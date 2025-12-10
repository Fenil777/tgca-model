"""Evaluation metrics for survival prediction."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_concordance_index(
    event_times: np.ndarray,
    predicted_risk: np.ndarray,
    event_observed: np.ndarray,
) -> float:
    """
    Compute Harrell's concordance index (C-index).
    
    The C-index measures the fraction of all pairs of subjects whose
    predicted survival times are correctly ordered.
    
    Args:
        event_times: Actual survival times
        predicted_risk: Predicted risk scores (higher = higher risk)
        event_observed: Binary event indicators (1=event, 0=censored)
        
    Returns:
        C-index value (0.5 = random, 1.0 = perfect)
    """
    logger.info("Computing concordance index")
    
    c_index = concordance_index(
        event_times=event_times,
        predicted_scores=predicted_risk,
        event_observed=event_observed
    )
    
    logger.info(f"C-index: {c_index:.4f}")
    return c_index


def compute_brier_score(
    event_times: np.ndarray,
    survival_probs: np.ndarray,
    event_observed: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    Compute time-dependent Brier score (placeholder).
    
    The Brier score measures the accuracy of probabilistic predictions.
    Lower is better (0 = perfect).
    
    Note: This is a placeholder implementation. For production use, implement proper
    Brier score calculation using Inverse Probability of Censoring Weighting (IPCW):
    
    .. code-block:: python
    
        from sksurv.metrics import brier_score
        # or
        from lifelines.utils import concordance_index
    
    Reference:
        Graf et al. (1999). Assessment and comparison of prognostic 
        classification schemes for survival data.
    
    Args:
        event_times: Actual survival times
        survival_probs: Predicted survival probabilities at each time point
        event_observed: Binary event indicators
        times: Time points at which to compute Brier score
        
    Returns:
        Brier scores at each time point (placeholder - not actual scores)
        
    Raises:
        NotImplementedError: Always raised to indicate this is a placeholder
    """
    logger.error("compute_brier_score is a placeholder and not implemented")
    raise NotImplementedError(
        "Brier score calculation is not implemented. "
        "Use sksurv.metrics.brier_score or implement IPCW-based Brier score. "
        "See: Graf et al. (1999) for methodology."
    )


def compute_integrated_brier_score(
    event_times: np.ndarray,
    survival_probs: np.ndarray,
    event_observed: np.ndarray,
    times: np.ndarray,
) -> float:
    """
    Compute integrated Brier score.
    
    **Note: This function is not implemented.** It raises NotImplementedError
    because it depends on compute_brier_score which is not implemented.
    
    For production implementation, use:
    - sksurv.metrics.integrated_brier_score (requires scikit-survival)
    - Or compute Brier scores at multiple time points and integrate
    
    Args:
        event_times: Actual survival times
        survival_probs: Predicted survival probabilities
        event_observed: Binary event indicators
        times: Time points for integration
        
    Returns:
        Integrated Brier score
        
    Raises:
        NotImplementedError: Always raised because compute_brier_score is not implemented
    """
    logger.error("compute_integrated_brier_score is not implemented")
    raise NotImplementedError(
        "Integrated Brier score calculation is not implemented. "
        "Use sksurv.metrics.integrated_brier_score or implement based on IPCW Brier scores."
    )


def evaluate_model(
    model_name: str,
    y_true: pd.DataFrame,
    y_pred_risk: np.ndarray,
    time_col: str = "OS_time",
    event_col: str = "OS_status",
) -> dict:
    """
    Evaluate survival model with multiple metrics.
    
    Args:
        model_name: Name of the model
        y_true: True survival data (time and event)
        y_pred_risk: Predicted risk scores
        time_col: Name of time column
        event_col: Name of event column
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating model: {model_name}")
    
    # Concordance index
    c_index = compute_concordance_index(
        event_times=y_true[time_col].values,
        predicted_risk=y_pred_risk,
        event_observed=y_true[event_col].values
    )
    
    # Summary statistics
    n_samples = len(y_true)
    n_events = y_true[event_col].sum()
    event_rate = n_events / n_samples
    
    metrics = {
        "model": model_name,
        "c_index": c_index,
        "n_samples": n_samples,
        "n_events": int(n_events),
        "event_rate": event_rate,
    }
    
    logger.info(f"Evaluation complete: C-index = {c_index:.4f}")
    return metrics


def compare_models(
    results: list,
) -> pd.DataFrame:
    """
    Compare multiple model results.
    
    Args:
        results: List of evaluation result dictionaries
        
    Returns:
        DataFrame with comparison
    """
    logger.info(f"Comparing {len(results)} models")
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values("c_index", ascending=False)
    
    return comparison_df


def stratify_by_risk(
    risk_scores: np.ndarray,
    n_groups: int = 2,
    labels: Optional[list] = None,
) -> np.ndarray:
    """
    Stratify patients into risk groups based on risk scores.
    
    Args:
        risk_scores: Predicted risk scores
        n_groups: Number of risk groups
        labels: Group labels (default: ['Low', 'High'] for 2 groups)
        
    Returns:
        Array of group assignments
    """
    logger.info(f"Stratifying into {n_groups} risk groups")
    
    if labels is None:
        if n_groups == 2:
            labels = ["Low Risk", "High Risk"]
        elif n_groups == 3:
            labels = ["Low Risk", "Medium Risk", "High Risk"]
        else:
            labels = [f"Group {i+1}" for i in range(n_groups)]
    
    # Compute quantiles for stratification
    quantiles = np.linspace(0, 100, n_groups + 1)
    thresholds = np.percentile(risk_scores, quantiles[1:-1])
    
    # Assign groups
    groups = np.digitize(risk_scores, thresholds)
    group_labels = np.array([labels[i] for i in groups])
    
    # Log distribution
    unique, counts = np.unique(group_labels, return_counts=True)
    for label, count in zip(unique, counts):
        logger.info(f"  {label}: {count} samples ({count/len(risk_scores)*100:.1f}%)")
    
    return group_labels


def compute_survival_metrics_at_time(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    survival_probs: np.ndarray,
    time_point: float,
    threshold: float = 0.5,
) -> dict:
    """
    Compute classification metrics at a specific time point.
    
    Treat survival prediction as a binary classification problem at time t:
    - Positive: survives beyond time t
    - Negative: event occurs before time t
    
    Args:
        event_times: Actual survival times
        event_observed: Binary event indicators
        survival_probs: Predicted survival probabilities at time_point
        time_point: Time point for evaluation
        threshold: Probability threshold for classification
        
    Returns:
        Dictionary with metrics (sensitivity, specificity, etc.)
    """
    logger.info(f"Computing metrics at time point: {time_point}")
    
    # True labels: survived beyond time_point
    y_true = (event_times > time_point) | ((event_times <= time_point) & (event_observed == 0))
    
    # Predicted labels
    y_pred = survival_probs > threshold
    
    # Compute metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    metrics = {
        "time_point": time_point,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    
    return metrics


if __name__ == "__main__":
    # Example usage
    n_samples = 100
    
    # Generate sample data
    event_times = np.random.randint(100, 1000, n_samples)
    event_observed = np.random.choice([0, 1], n_samples)
    predicted_risk = np.random.randn(n_samples)
    
    # Compute C-index
    c_index = compute_concordance_index(event_times, predicted_risk, event_observed)
    print(f"C-index: {c_index:.4f}")
    
    # Stratify by risk
    groups = stratify_by_risk(predicted_risk, n_groups=2)
    print(f"\nRisk groups: {np.unique(groups)}")
