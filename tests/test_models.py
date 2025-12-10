"""Tests for survival models."""

import pytest
import numpy as np
import pandas as pd

from src.models.cox_model import train_cox_model, predict_risk, evaluate_cox_model
from src.models.deepsurv_model import train_deepsurv_model, predict_risk_deepsurv


class TestCoxModel:
    """Test Cox Proportional Hazards model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n_samples = 100
        n_features = 20
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        y = pd.DataFrame({
            "OS_time": np.random.randint(100, 1000, n_samples),
            "OS_status": np.random.choice([0, 1], n_samples)
        })
        
        return X, y
    
    def test_train_cox_model(self, sample_data):
        """Test Cox model training."""
        X, y = sample_data
        
        model = train_cox_model(X, y)
        
        assert model is not None
        assert hasattr(model, "predict_partial_hazard")
    
    def test_predict_risk(self, sample_data):
        """Test risk prediction."""
        X, y = sample_data
        
        model = train_cox_model(X, y)
        risk_scores = predict_risk(model, X)
        
        assert isinstance(risk_scores, np.ndarray)
        assert len(risk_scores) == len(X)
        assert not np.isnan(risk_scores).any()
    
    def test_evaluate_cox_model(self, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        model = train_cox_model(X_train, y_train)
        metrics = evaluate_cox_model(model, X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert "c_index" in metrics
        assert 0 <= metrics["c_index"] <= 1


class TestRSFModel:
    """Test Random Survival Forest model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n_samples = 100
        n_features = 20
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        y = pd.DataFrame({
            "OS_time": np.random.randint(100, 1000, n_samples),
            "OS_status": np.random.choice([0, 1], n_samples)
        })
        
        return X, y
    
    def test_train_rsf_model(self, sample_data):
        """Test RSF model training (requires scikit-survival)."""
        X, y = sample_data
        
        try:
            from src.models.rsf_model import train_rsf_model, evaluate_rsf_model
            
            model = train_rsf_model(X, y)
            
            assert model is not None
            assert hasattr(model, "predict")
            
            # Test evaluation
            X_train, X_test = X[:80], X[80:]
            y_train, y_test = y[:80], y[80:]
            
            model = train_rsf_model(X_train, y_train)
            metrics = evaluate_rsf_model(model, X_test, y_test)
            
            assert isinstance(metrics, dict)
            assert "c_index" in metrics
            
        except ImportError:
            pytest.skip("scikit-survival not installed")


class TestDeepSurvModel:
    """Test DeepSurv model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n_samples = 100
        n_features = 20
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        y = pd.DataFrame({
            "OS_time": np.random.randint(100, 1000, n_samples),
            "OS_status": np.random.choice([0, 1], n_samples)
        })
        
        return X, y
    
    def test_train_deepsurv_model(self, sample_data):
        """Test DeepSurv model training."""
        X, y = sample_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        
        model, trainer = train_deepsurv_model(
            X_train, y_train,
            X_val, y_val,
            hidden_dims=[16, 8],
            num_epochs=5,
            batch_size=16
        )
        
        assert model is not None
        assert trainer is not None
        assert hasattr(trainer, "predict")
    
    def test_predict_risk_deepsurv(self, sample_data):
        """Test DeepSurv risk prediction."""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        model, trainer = train_deepsurv_model(
            X_train, y_train,
            hidden_dims=[16, 8],
            num_epochs=5,
            batch_size=16
        )
        
        risk_scores = predict_risk_deepsurv(trainer, X_test)
        
        assert isinstance(risk_scores, np.ndarray)
        assert len(risk_scores) == len(X_test)
        assert not np.isnan(risk_scores).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
