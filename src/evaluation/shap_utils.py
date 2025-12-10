"""SHAP utilities for model interpretability."""

from typing import Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def check_shap_available() -> bool:
    """Check if SHAP is installed."""
    try:
        import shap
        return True
    except ImportError:
        return False


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    model_type: str = "tree",
    background_samples: Optional[int] = 100,
) -> Any:
    """
    Compute SHAP values for model predictions.
    
    Args:
        model: Trained model
        X: Features for explanation
        model_type: Type of explainer ('tree', 'kernel', 'linear')
        background_samples: Number of background samples for KernelExplainer
        
    Returns:
        SHAP values
        
    Raises:
        ImportError: If SHAP is not installed
    """
    if not check_shap_available():
        raise ImportError(
            "SHAP is required for interpretability. "
            "Install it with: pip install shap"
        )
    
    import shap
    
    logger.info(f"Computing SHAP values using {model_type} explainer")
    
    try:
        if model_type == "tree":
            # TreeExplainer for tree-based models (RSF, GBM, etc.)
            logger.info("Using TreeExplainer")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
        elif model_type == "kernel":
            # KernelExplainer for any model (slower but universal)
            logger.info("Using KernelExplainer")
            logger.warning("KernelExplainer can be slow for large datasets")
            
            # Sample background data
            if len(X) > background_samples:
                background = X.sample(n=background_samples, random_state=42)
            else:
                background = X
            
            # Define prediction function
            def predict_fn(data):
                if hasattr(model, 'predict_partial_hazard'):
                    # CoxPH model
                    return model.predict_partial_hazard(pd.DataFrame(data, columns=X.columns)).values
                elif hasattr(model, 'predict'):
                    # Other models
                    return model.predict(data)
                else:
                    raise ValueError("Model does not have predict method")
            
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X)
            
        elif model_type == "linear":
            # LinearExplainer for linear models
            logger.info("Using LinearExplainer")
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X)
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        logger.info(f"Computed SHAP values for {len(X)} samples")
        return shap_values
        
    except Exception as e:
        logger.error(f"Error computing SHAP values: {e}")
        logger.info("Falling back to KernelExplainer")
        
        # Fallback to KernelExplainer
        if len(X) > background_samples:
            background = X.sample(n=background_samples, random_state=42)
        else:
            background = X
        
        def predict_fn(data):
            if hasattr(model, 'predict_partial_hazard'):
                return model.predict_partial_hazard(pd.DataFrame(data, columns=X.columns)).values
            elif hasattr(model, 'predict'):
                return model.predict(data)
            else:
                raise ValueError("Model does not have predict method")
        
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X)
        
        return shap_values


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_display: int = 20,
    title: str = "SHAP Feature Importance",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot SHAP summary plot.
    
    Args:
        shap_values: SHAP values
        X: Features
        max_display: Maximum number of features to display
        title: Plot title
        save_path: Path to save figure (optional)
        
    Raises:
        ImportError: If SHAP is not installed
    """
    if not check_shap_available():
        raise ImportError("SHAP is required")
    
    import shap
    
    logger.info("Plotting SHAP summary")
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X,
        max_display=max_display,
        show=False
    )
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    sample_idx: int = 0,
    max_display: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot SHAP waterfall plot for a single sample.
    
    Args:
        shap_values: SHAP values
        X: Features
        sample_idx: Index of sample to explain
        max_display: Maximum number of features to display
        save_path: Path to save figure (optional)
        
    Raises:
        ImportError: If SHAP is not installed
    """
    if not check_shap_available():
        raise ImportError("SHAP is required")
    
    import shap
    
    logger.info(f"Plotting SHAP waterfall for sample {sample_idx}")
    
    plt.figure(figsize=(10, 6))
    
    # Create explanation object - handle different SHAP versions
    try:
        if isinstance(shap_values, np.ndarray):
            # For older SHAP versions or numpy arrays
            # Note: waterfall_legacy is a private API and may change in future versions
            try:
                shap.plots._waterfall.waterfall_legacy(
                    shap_values[sample_idx],
                    max_display=max_display
                )
            except AttributeError:
                # If private API changed, fall back to summary plot
                logger.warning("waterfall_legacy not available, using summary plot")
                shap.summary_plot(
                    shap_values[sample_idx:sample_idx+1],
                    X.iloc[sample_idx:sample_idx+1],
                    max_display=max_display,
                    show=False
                )
        else:
            # For newer SHAP versions with Explanation objects (public API)
            shap.plots.waterfall(shap_values[sample_idx], max_display=max_display)
    except Exception as e:
        logger.error(f"Error creating waterfall plot: {e}")
        logger.info("Falling back to bar plot")
        # Fallback: simple bar plot
        values = shap_values[sample_idx] if isinstance(shap_values, np.ndarray) else shap_values.values[sample_idx]
        features = X.columns.tolist()
        sorted_idx = np.argsort(np.abs(values))[-max_display:]
        plt.barh(range(len(sorted_idx)), values[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
        plt.xlabel('SHAP value')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot SHAP dependence plot for a specific feature.
    
    Args:
        shap_values: SHAP values
        X: Features
        feature: Feature to plot
        interaction_feature: Feature to use for coloring (optional)
        save_path: Path to save figure (optional)
        
    Raises:
        ImportError: If SHAP is not installed
    """
    if not check_shap_available():
        raise ImportError("SHAP is required")
    
    import shap
    
    logger.info(f"Plotting SHAP dependence for {feature}")
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature,
        shap_values,
        X,
        interaction_index=interaction_feature,
        show=False
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def get_top_shap_features(
    shap_values: np.ndarray,
    feature_names: list,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Get top features by mean absolute SHAP value.
    
    Args:
        shap_values: SHAP values
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with top features and their importance
    """
    logger.info(f"Getting top {top_n} SHAP features")
    
    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    })
    
    # Sort and select top N
    importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
    top_features = importance_df.head(top_n)
    
    logger.info(f"Top feature: {top_features.iloc[0]['feature']} "
                f"(SHAP: {top_features.iloc[0]['mean_abs_shap']:.4f})")
    
    return top_features


def explain_prediction(
    model: Any,
    X_sample: pd.DataFrame,
    X_background: pd.DataFrame,
    model_type: str = "kernel",
) -> dict:
    """
    Explain a single prediction with SHAP.
    
    Args:
        model: Trained model
        X_sample: Single sample to explain (1 x n_features)
        X_background: Background data for KernelExplainer
        model_type: Type of explainer
        
    Returns:
        Dictionary with explanation
        
    Raises:
        ImportError: If SHAP is not installed
    """
    if not check_shap_available():
        raise ImportError("SHAP is required")
    
    logger.info("Explaining single prediction")
    
    # Compute SHAP values
    shap_values = compute_shap_values(
        model,
        X_sample,
        model_type=model_type,
        background_samples=min(100, len(X_background))
    )
    
    # Get feature contributions
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    contributions = pd.DataFrame({
        'feature': X_sample.columns,
        'value': X_sample.iloc[0].values,
        'shap_value': shap_values[0] if len(shap_values.shape) > 1 else shap_values
    })
    contributions = contributions.sort_values('shap_value', key=abs, ascending=False)
    
    explanation = {
        'contributions': contributions,
        'base_value': shap_values.mean() if hasattr(shap_values, 'mean') else 0,
        'prediction': contributions['shap_value'].sum(),
    }
    
    return explanation


if __name__ == "__main__":
    # Example usage (requires SHAP and a trained model)
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier
        
        # Create sample data
        X = pd.DataFrame(
            np.random.randn(100, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        y = np.random.choice([0, 1], 100)
        
        # Train simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Compute SHAP values
        shap_values = compute_shap_values(model, X, model_type="tree")
        
        # Plot summary
        plot_shap_summary(shap_values, X)
        
        # Get top features
        top_features = get_top_shap_features(shap_values, X.columns.tolist(), top_n=5)
        print("\nTop 5 features:")
        print(top_features)
        
    except ImportError as e:
        print(f"Cannot run example: {e}")
        print("Install SHAP to use interpretability features")
