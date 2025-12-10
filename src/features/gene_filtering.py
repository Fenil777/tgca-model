"""Gene filtering and selection utilities."""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from ..config import GENE_FILTERING_CONFIG
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def select_variable_genes(
    expression_df: pd.DataFrame,
    top_n: Optional[int] = None,
    variance_threshold: Optional[float] = None,
    patient_id_col: str = "patient_id",
) -> List[str]:
    """
    Select genes with highest variance.
    
    Args:
        expression_df: DataFrame with gene expression data
        top_n: Number of top variable genes to select
        variance_threshold: Minimum variance threshold
        patient_id_col: Name of patient ID column to exclude
        
    Returns:
        List of selected gene names
    """
    logger.info("Selecting variable genes")
    
    top_n = top_n or GENE_FILTERING_CONFIG["top_n_genes"]
    variance_threshold = variance_threshold or GENE_FILTERING_CONFIG["variance_threshold"]
    
    # Get gene columns (exclude patient_id and survival columns)
    gene_cols = [c for c in expression_df.columns 
                if c != patient_id_col and c not in ['OS_time', 'OS_status']]
    
    logger.info(f"Starting with {len(gene_cols)} genes")
    
    # Calculate variance for each gene
    gene_data = expression_df[gene_cols]
    gene_variances = gene_data.var()
    
    # Filter by variance threshold
    high_var_genes = gene_variances[gene_variances > variance_threshold].index.tolist()
    logger.info(f"After variance filtering (>{variance_threshold}): {len(high_var_genes)} genes")
    
    # Select top N by variance
    if len(high_var_genes) > top_n:
        top_var_genes = gene_variances.nlargest(top_n).index.tolist()
        logger.info(f"Selected top {top_n} genes by variance")
    else:
        top_var_genes = high_var_genes
        logger.info(f"Using all {len(top_var_genes)} variable genes")
    
    return top_var_genes


def filter_low_expression(
    expression_df: pd.DataFrame,
    min_expression: Optional[float] = None,
    min_samples: int = 10,
    patient_id_col: str = "patient_id",
) -> List[str]:
    """
    Filter genes with low expression across samples.
    
    Args:
        expression_df: DataFrame with gene expression data
        min_expression: Minimum expression threshold
        min_samples: Minimum number of samples above threshold
        patient_id_col: Name of patient ID column to exclude
        
    Returns:
        List of filtered gene names
    """
    logger.info("Filtering low expression genes")
    
    min_expression = min_expression or GENE_FILTERING_CONFIG["min_expression"]
    
    # Get gene columns
    gene_cols = [c for c in expression_df.columns 
                if c != patient_id_col and c not in ['OS_time', 'OS_status']]
    
    gene_data = expression_df[gene_cols]
    
    # Count samples with expression above threshold
    samples_above_threshold = (gene_data > min_expression).sum()
    
    # Filter genes
    filtered_genes = samples_above_threshold[samples_above_threshold >= min_samples].index.tolist()
    
    logger.info(f"After expression filtering: {len(filtered_genes)} genes")
    logger.info(f"Removed {len(gene_cols) - len(filtered_genes)} low-expression genes")
    
    return filtered_genes


def select_genes_by_survival(
    expression_df: pd.DataFrame,
    time_col: str = "OS_time",
    event_col: str = "OS_status",
    top_n: int = 100,
    method: str = "univariate",
) -> List[str]:
    """
    Select genes associated with survival (placeholder for univariate Cox).
    
    This is a placeholder. In practice, you would:
    - Fit univariate Cox models for each gene
    - Rank by p-value or concordance index
    - Select top N associated genes
    
    Args:
        expression_df: DataFrame with expression and survival data
        time_col: Survival time column
        event_col: Event status column
        top_n: Number of genes to select
        method: Selection method ('univariate')
        
    Returns:
        List of selected gene names
    """
    logger.info(f"Selecting genes by survival association (method: {method})")
    
    # Get gene columns
    gene_cols = [c for c in expression_df.columns 
                if c not in ['patient_id', time_col, event_col]]
    
    if len(gene_cols) <= top_n:
        logger.warning(f"Fewer genes ({len(gene_cols)}) than requested ({top_n})")
        return gene_cols
    
    # TODO: Implement proper survival-based gene selection
    # In production, fit univariate Cox models for each gene and rank by:
    # - p-value (smaller is better)
    # - concordance index (higher is better)
    # - hazard ratio significance
    # Example implementation would use:
    # from lifelines import CoxPHFitter
    # for gene in gene_cols:
    #     cph = CoxPHFitter()
    #     cph.fit(pd.DataFrame({gene: expression_df[gene], 
    #                           'OS_time': expression_df[time_col],
    #                           'OS_status': expression_df[event_col]}),
    #            duration_col=time_col, event_col=event_col)
    #     p_values[gene] = cph.summary['p'].values[0]
    # selected_genes = sorted(p_values, key=p_values.get)[:top_n]
    
    # Placeholder: random selection (replace with above logic for production)
    logger.warning("Using random selection as placeholder for survival-based selection. "
                   "Replace with univariate Cox regression for production.")
    selected_genes = np.random.choice(gene_cols, size=top_n, replace=False).tolist()
    
    logger.info(f"Selected {len(selected_genes)} genes")
    return selected_genes


def apply_gene_filter(
    df: pd.DataFrame,
    selected_genes: List[str],
    keep_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply gene filter to keep only selected genes.
    
    Args:
        df: DataFrame with all genes
        selected_genes: List of genes to keep
        keep_cols: Additional columns to keep (e.g., patient_id, survival)
        
    Returns:
        Filtered DataFrame
    """
    keep_cols = keep_cols or ['patient_id', 'OS_time', 'OS_status']
    
    # Keep only selected genes and specified columns
    cols_to_keep = [c for c in keep_cols if c in df.columns] + selected_genes
    filtered_df = df[cols_to_keep].copy()
    
    logger.info(f"Filtered DataFrame shape: {filtered_df.shape}")
    return filtered_df


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample data
    n_samples = 100
    n_genes = 500
    
    data = pd.DataFrame(
        np.random.randn(n_samples, n_genes),
        columns=[f'GENE{i}' for i in range(n_genes)]
    )
    data['patient_id'] = [f'P{i}' for i in range(n_samples)]
    data['OS_time'] = np.random.randint(100, 1000, n_samples)
    data['OS_status'] = np.random.choice([0, 1], n_samples)
    
    # Select variable genes
    selected = select_variable_genes(data, top_n=50)
    print(f"Selected {len(selected)} genes")
    
    # Apply filter
    filtered = apply_gene_filter(data, selected)
    print(f"Filtered shape: {filtered.shape}")
