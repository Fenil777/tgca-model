"""Tests for evaluation metrics and plotting."""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation.metrics import (
    compute_concordance_index,
    stratify_by_risk,
    evaluate_model,
)
from src.evaluation.survival_plots import (
    plot_kaplan_meier,
    plot_risk_distribution,
)


class TestMetrics:
    """Test evaluation metrics."""
    
    @pytest.fixture
    def sample_survival_data(self):
        """Create sample survival data."""
        n_samples = 100
        
        event_times = np.random.randint(100, 1000, n_samples)
        event_observed = np.random.choice([0, 1], n_samples)
        predicted_risk = np.random.randn(n_samples)
        
        return event_times, event_observed, predicted_risk
    
    def test_compute_concordance_index(self, sample_survival_data):
        """Test C-index computation."""
        event_times, event_observed, predicted_risk = sample_survival_data
        
        c_index = compute_concordance_index(
            event_times, predicted_risk, event_observed
        )
        
        assert isinstance(c_index, float)
        assert 0 <= c_index <= 1
    
    def test_stratify_by_risk(self, sample_survival_data):
        """Test risk stratification."""
        _, _, predicted_risk = sample_survival_data
        
        # Test 2 groups
        groups = stratify_by_risk(predicted_risk, n_groups=2)
        
        assert isinstance(groups, np.ndarray)
        assert len(groups) == len(predicted_risk)
        assert len(np.unique(groups)) == 2
        
        # Test 3 groups
        groups = stratify_by_risk(predicted_risk, n_groups=3)
        
        assert len(np.unique(groups)) == 3
    
    def test_evaluate_model(self, sample_survival_data):
        """Test model evaluation wrapper."""
        event_times, event_observed, predicted_risk = sample_survival_data
        
        y_true = pd.DataFrame({
            "OS_time": event_times,
            "OS_status": event_observed
        })
        
        metrics = evaluate_model("test_model", y_true, predicted_risk)
        
        assert isinstance(metrics, dict)
        assert "model" in metrics
        assert "c_index" in metrics
        assert "n_samples" in metrics
        assert metrics["n_samples"] == len(event_times)


class TestSurvivalPlots:
    """Test survival plotting functions."""
    
    @pytest.fixture
    def sample_survival_data(self):
        """Create sample survival data."""
        n_samples = 100
        
        event_times = np.random.randint(100, 1000, n_samples)
        event_observed = np.random.choice([0, 1], n_samples)
        predicted_risk = np.random.randn(n_samples)
        
        return event_times, event_observed, predicted_risk
    
    def test_plot_kaplan_meier(self, sample_survival_data):
        """Test Kaplan-Meier plotting."""
        event_times, event_observed, _ = sample_survival_data
        
        fig = plot_kaplan_meier(event_times, event_observed)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_kaplan_meier_with_groups(self, sample_survival_data):
        """Test Kaplan-Meier plotting with groups."""
        event_times, event_observed, predicted_risk = sample_survival_data
        
        groups = (predicted_risk > np.median(predicted_risk)).astype(int)
        
        fig = plot_kaplan_meier(
            event_times,
            event_observed,
            groups=groups,
            group_labels=["Low Risk", "High Risk"]
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_risk_distribution(self, sample_survival_data):
        """Test risk distribution plotting."""
        _, event_observed, predicted_risk = sample_survival_data
        
        fig = plot_risk_distribution(predicted_risk, event_observed)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestFeatureEngineering:
    """Test feature engineering functions."""
    
    def test_gene_filtering(self):
        """Test gene filtering."""
        from src.features.gene_filtering import select_variable_genes
        
        n_samples = 50
        n_genes = 100
        
        df = pd.DataFrame(
            np.random.randn(n_samples, n_genes),
            columns=[f"GENE{i}" for i in range(n_genes)]
        )
        df["patient_id"] = [f"P{i}" for i in range(n_samples)]
        df["OS_time"] = np.random.randint(100, 1000, n_samples)
        df["OS_status"] = np.random.choice([0, 1], n_samples)
        
        selected_genes = select_variable_genes(df, top_n=20)
        
        assert isinstance(selected_genes, list)
        assert len(selected_genes) <= 20
        assert all(gene.startswith("GENE") for gene in selected_genes)
    
    def test_pathway_aggregation(self):
        """Test pathway score computation."""
        from src.features.pathway_aggregation import compute_pathway_scores
        
        n_samples = 50
        genes = ["PIK3CA", "AKT1", "PTEN", "KRAS", "TP53", "GENE1"]
        
        df = pd.DataFrame(
            np.random.randn(n_samples, len(genes)),
            columns=genes
        )
        df["patient_id"] = [f"P{i}" for i in range(n_samples)]
        
        pathway_scores = compute_pathway_scores(df)
        
        assert isinstance(pathway_scores, pd.DataFrame)
        assert len(pathway_scores) == len(df)
        assert any(col.startswith("pathway_") for col in pathway_scores.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
