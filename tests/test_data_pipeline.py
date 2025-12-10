"""Tests for data processing pipeline."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.download_tcga import create_sample_data
from src.data.load_tcga import (
    load_clinical_data,
    load_expression_data,
    merge_clinical_expression,
    validate_survival_data,
)
from src.data.preprocess import (
    clean_clinical_features,
    normalize_expression,
    split_train_test,
)
from src.config import RAW_DATA_DIR


class TestDataDownload:
    """Test data download and sample generation."""
    
    def test_create_sample_data(self):
        """Test sample data generation."""
        create_sample_data("BRCA")
        
        clinical_path = RAW_DATA_DIR / "BRCA_clinical.csv"
        expression_path = RAW_DATA_DIR / "BRCA_expression.csv"
        
        assert clinical_path.exists()
        assert expression_path.exists()


class TestDataLoading:
    """Test data loading functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        create_sample_data("BRCA")
        return {
            "clinical": RAW_DATA_DIR / "BRCA_clinical.csv",
            "expression": RAW_DATA_DIR / "BRCA_expression.csv",
        }
    
    def test_load_clinical_data(self, sample_data):
        """Test clinical data loading."""
        df = load_clinical_data(sample_data["clinical"])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "patient_id" in df.columns
        assert "OS_time" in df.columns
        assert "OS_status" in df.columns
    
    def test_load_expression_data(self, sample_data):
        """Test expression data loading."""
        df = load_expression_data(sample_data["expression"])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "patient_id" in df.columns
    
    def test_merge_clinical_expression(self, sample_data):
        """Test merging clinical and expression data."""
        clinical_df = load_clinical_data(sample_data["clinical"])
        expression_df = load_expression_data(sample_data["expression"])
        
        merged_df = merge_clinical_expression(clinical_df, expression_df)
        
        assert isinstance(merged_df, pd.DataFrame)
        assert len(merged_df) > 0
        assert "OS_time" in merged_df.columns
        assert "patient_id" in merged_df.columns
    
    def test_validate_survival_data(self, sample_data):
        """Test survival data validation."""
        clinical_df = load_clinical_data(sample_data["clinical"])
        expression_df = load_expression_data(sample_data["expression"])
        merged_df = merge_clinical_expression(clinical_df, expression_df)
        
        validated_df = validate_survival_data(merged_df)
        
        assert isinstance(validated_df, pd.DataFrame)
        assert len(validated_df) > 0
        assert validated_df["OS_time"].min() >= 0
        assert set(validated_df["OS_status"].unique()).issubset({0, 1})


class TestPreprocessing:
    """Test preprocessing functions."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        n_samples = 50
        n_genes = 20
        
        df = pd.DataFrame(
            np.random.randn(n_samples, n_genes),
            columns=[f"GENE{i}" for i in range(n_genes)]
        )
        df["patient_id"] = [f"P{i}" for i in range(n_samples)]
        df["age_at_diagnosis"] = np.random.randint(30, 80, n_samples)
        df["gender"] = np.random.choice(["Male", "Female"], n_samples)
        df["tumor_stage"] = np.random.choice(["Stage I", "Stage II"], n_samples)
        df["OS_time"] = np.random.randint(100, 1000, n_samples)
        df["OS_status"] = np.random.choice([0, 1], n_samples)
        
        return df
    
    def test_clean_clinical_features(self, sample_df):
        """Test clinical feature cleaning."""
        cleaned_df = clean_clinical_features(sample_df)
        
        assert isinstance(cleaned_df, pd.DataFrame)
        assert len(cleaned_df) == len(sample_df)
        # Check that categorical variables were encoded
        assert any("gender_" in col for col in cleaned_df.columns)
    
    def test_normalize_expression(self, sample_df):
        """Test expression normalization."""
        normalized_df, scaler = normalize_expression(sample_df)
        
        assert isinstance(normalized_df, pd.DataFrame)
        assert len(normalized_df) == len(sample_df)
        assert scaler is not None
        
        # Check that gene columns were normalized
        gene_cols = [c for c in sample_df.columns if c.startswith("GENE")]
        for col in gene_cols:
            assert col in normalized_df.columns
    
    def test_split_train_test(self, sample_df):
        """Test train/test splitting."""
        train_df, test_df = split_train_test(sample_df, test_size=0.2)
        
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert len(train_df) + len(test_df) == len(sample_df)
        assert len(train_df) > len(test_df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
