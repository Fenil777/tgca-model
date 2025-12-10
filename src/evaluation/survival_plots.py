"""Visualization utilities for survival analysis."""

from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_kaplan_meier(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    groups: Optional[np.ndarray] = None,
    group_labels: Optional[List[str]] = None,
    title: str = "Kaplan-Meier Survival Curves",
    xlabel: str = "Time (days)",
    ylabel: str = "Survival Probability",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Kaplan-Meier survival curves.
    
    Args:
        event_times: Survival times
        event_observed: Event indicators
        groups: Group assignments for stratification (optional)
        group_labels: Labels for groups (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting Kaplan-Meier curves")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if groups is None:
        # Single curve
        kmf = KaplanMeierFitter()
        kmf.fit(event_times, event_observed, label="All patients")
        kmf.plot_survival_function(ax=ax)
    else:
        # Multiple curves (one per group)
        unique_groups = np.unique(groups)
        
        if group_labels is None:
            group_labels = [f"Group {g}" for g in unique_groups]
        
        kmfs = []
        for i, group in enumerate(unique_groups):
            mask = groups == group
            kmf = KaplanMeierFitter()
            kmf.fit(
                event_times[mask],
                event_observed[mask],
                label=group_labels[i] if i < len(group_labels) else f"Group {group}"
            )
            kmf.plot_survival_function(ax=ax)
            kmfs.append(kmf)
        
        # Add at-risk counts
        try:
            add_at_risk_counts(*kmfs, ax=ax)
        except:
            logger.warning("Could not add at-risk counts")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    return fig


def plot_survival_function(
    times: np.ndarray,
    survival_probs: np.ndarray,
    patient_ids: Optional[List[str]] = None,
    title: str = "Predicted Survival Functions",
    xlabel: str = "Time (days)",
    ylabel: str = "Survival Probability",
    max_patients: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot predicted survival functions for individual patients.
    
    Args:
        times: Time points
        survival_probs: Survival probabilities (n_patients, n_times)
        patient_ids: Patient identifiers (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        max_patients: Maximum number of patients to plot
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting survival functions for up to {max_patients} patients")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Limit number of patients to plot
    n_patients = min(survival_probs.shape[0], max_patients)
    
    for i in range(n_patients):
        label = patient_ids[i] if patient_ids and i < len(patient_ids) else f"Patient {i+1}"
        ax.plot(times, survival_probs[i], label=label, alpha=0.7)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    return fig


def plot_risk_distribution(
    risk_scores: np.ndarray,
    event_observed: Optional[np.ndarray] = None,
    title: str = "Risk Score Distribution",
    xlabel: str = "Risk Score",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot distribution of risk scores.
    
    Args:
        risk_scores: Predicted risk scores
        event_observed: Event indicators (optional, for coloring)
        title: Plot title
        xlabel: X-axis label
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting risk score distribution")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if event_observed is not None:
        # Separate distributions for events vs censored
        ax.hist(
            risk_scores[event_observed == 1],
            bins=30,
            alpha=0.5,
            label="Event",
            color="red"
        )
        ax.hist(
            risk_scores[event_observed == 0],
            bins=30,
            alpha=0.5,
            label="Censored",
            color="blue"
        )
        ax.legend()
    else:
        ax.hist(risk_scores, bins=30, alpha=0.7, color="steelblue")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    return fig


def plot_calibration(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    predicted_probs: np.ndarray,
    time_point: float,
    n_bins: int = 10,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot calibration curve at a specific time point.
    
    Args:
        event_times: Actual survival times
        event_observed: Event indicators
        predicted_probs: Predicted survival probabilities at time_point
        time_point: Time point for calibration
        n_bins: Number of bins for calibration
        title: Plot title (optional)
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting calibration curve at time {time_point}")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Compute observed survival at time_point
    observed = (event_times > time_point) | ((event_times <= time_point) & (event_observed == 0))
    
    # Bin predictions
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predicted_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Compute mean predicted and observed for each bin
    mean_predicted = []
    mean_observed = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_predicted.append(predicted_probs[mask].mean())
            mean_observed.append(observed[mask].mean())
    
    # Plot
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(mean_predicted, mean_observed, 'o-', label='Model calibration')
    
    ax.set_xlabel('Predicted Survival Probability')
    ax.set_ylabel('Observed Survival Proportion')
    ax.set_title(title or f'Calibration at {time_point} days')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importances: Feature importance values
        top_n: Number of top features to plot
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting top {top_n} features")
    
    # Sort and select top N
    indices = np.argsort(np.abs(importances))[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['red' if x < 0 else 'green' for x in top_importances]
    ax.barh(range(len(top_features)), top_importances, color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    n_samples = 200
    
    # Generate sample data
    event_times = np.random.randint(100, 1000, n_samples)
    event_observed = np.random.choice([0, 1], n_samples)
    risk_scores = np.random.randn(n_samples)
    
    # Stratify by risk
    groups = (risk_scores > np.median(risk_scores)).astype(int)
    
    # Plot Kaplan-Meier
    fig = plot_kaplan_meier(
        event_times,
        event_observed,
        groups=groups,
        group_labels=["Low Risk", "High Risk"]
    )
    plt.show()
    
    # Plot risk distribution
    fig = plot_risk_distribution(risk_scores, event_observed)
    plt.show()
