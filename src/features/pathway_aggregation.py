"""Pathway aggregation utilities."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import PATHWAY_GENES
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_pathway_scores(
    expression_df: pd.DataFrame,
    pathway_dict: Optional[Dict[str, List[str]]] = None,
    aggregation: str = "mean",
    patient_id_col: str = "patient_id",
) -> pd.DataFrame:
    """
    Compute pathway-level scores from gene expression.
    
    Args:
        expression_df: DataFrame with gene expression data
        pathway_dict: Dictionary mapping pathway names to gene lists
        aggregation: Aggregation method ('mean', 'median', 'sum')
        patient_id_col: Name of patient ID column
        
    Returns:
        DataFrame with pathway scores
    """
    logger.info(f"Computing pathway scores using {aggregation} aggregation")
    
    pathway_dict = pathway_dict or PATHWAY_GENES
    logger.info(f"Processing {len(pathway_dict)} pathways")
    
    # Get available genes in expression data
    gene_cols = [c for c in expression_df.columns 
                if c != patient_id_col and c not in ['OS_time', 'OS_status']]
    available_genes = set(gene_cols)
    
    pathway_scores = {}
    
    for pathway_name, pathway_genes in pathway_dict.items():
        # Find genes in this pathway that are in the data
        genes_in_data = [g for g in pathway_genes if g in available_genes]
        
        if not genes_in_data:
            logger.warning(f"No genes found for pathway {pathway_name}")
            continue
        
        logger.info(f"{pathway_name}: {len(genes_in_data)}/{len(pathway_genes)} genes available")
        
        # Aggregate expression across pathway genes
        pathway_expr = expression_df[genes_in_data]
        
        if aggregation == "mean":
            score = pathway_expr.mean(axis=1)
        elif aggregation == "median":
            score = pathway_expr.median(axis=1)
        elif aggregation == "sum":
            score = pathway_expr.sum(axis=1)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        pathway_scores[f"pathway_{pathway_name}"] = score
    
    # Create DataFrame
    pathway_df = pd.DataFrame(pathway_scores)
    
    # Add patient ID if available
    if patient_id_col in expression_df.columns:
        pathway_df.insert(0, patient_id_col, expression_df[patient_id_col].values)
    
    logger.info(f"Computed {len(pathway_scores)} pathway scores")
    return pathway_df


def add_pathway_features(
    expression_df: pd.DataFrame,
    pathway_dict: Optional[Dict[str, List[str]]] = None,
    aggregation: str = "mean",
) -> pd.DataFrame:
    """
    Add pathway scores as additional features to expression data.
    
    Args:
        expression_df: DataFrame with gene expression data
        pathway_dict: Dictionary mapping pathway names to gene lists
        aggregation: Aggregation method
        
    Returns:
        DataFrame with both gene expression and pathway scores
    """
    logger.info("Adding pathway features to expression data")
    
    pathway_scores = compute_pathway_scores(
        expression_df,
        pathway_dict=pathway_dict,
        aggregation=aggregation
    )
    
    # Merge pathway scores with expression data
    if 'patient_id' in expression_df.columns and 'patient_id' in pathway_scores.columns:
        merged_df = expression_df.merge(
            pathway_scores.drop(columns=['patient_id']),
            left_index=True,
            right_index=True
        )
    else:
        # If no patient_id, concatenate by index
        pathway_cols = [c for c in pathway_scores.columns if c != 'patient_id']
        merged_df = pd.concat([expression_df, pathway_scores[pathway_cols]], axis=1)
    
    logger.info(f"Combined shape: {merged_df.shape}")
    return merged_df


def compute_pathway_activation(
    expression_df: pd.DataFrame,
    pathway_dict: Optional[Dict[str, List[str]]] = None,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Compute binary pathway activation status.
    
    A pathway is considered "activated" if the mean expression of its genes
    is above the threshold.
    
    Args:
        expression_df: DataFrame with gene expression data
        pathway_dict: Dictionary mapping pathway names to gene lists
        threshold: Activation threshold
        
    Returns:
        DataFrame with binary pathway activation (0/1)
    """
    logger.info("Computing pathway activation status")
    
    # Compute pathway scores
    pathway_scores = compute_pathway_scores(
        expression_df,
        pathway_dict=pathway_dict,
        aggregation="mean"
    )
    
    # Convert to binary activation
    pathway_cols = [c for c in pathway_scores.columns if c.startswith('pathway_')]
    
    for col in pathway_cols:
        pathway_scores[f"{col}_active"] = (pathway_scores[col] > threshold).astype(int)
    
    logger.info(f"Computed activation status for {len(pathway_cols)} pathways")
    
    return pathway_scores


def get_pathway_genes(
    pathway_name: str,
    pathway_dict: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """
    Get list of genes for a specific pathway.
    
    Args:
        pathway_name: Name of the pathway
        pathway_dict: Dictionary mapping pathway names to gene lists
        
    Returns:
        List of gene names in the pathway
        
    Raises:
        KeyError: If pathway not found
    """
    pathway_dict = pathway_dict or PATHWAY_GENES
    
    if pathway_name not in pathway_dict:
        available = list(pathway_dict.keys())
        raise KeyError(
            f"Pathway '{pathway_name}' not found. "
            f"Available pathways: {available}"
        )
    
    return pathway_dict[pathway_name]


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample expression data
    n_samples = 100
    genes = ['PIK3CA', 'AKT1', 'PTEN', 'KRAS', 'TP53', 'GENE1', 'GENE2']
    
    data = pd.DataFrame(
        np.random.randn(n_samples, len(genes)),
        columns=genes
    )
    data['patient_id'] = [f'P{i}' for i in range(n_samples)]
    
    # Compute pathway scores
    pathway_scores = compute_pathway_scores(data)
    print(pathway_scores.head())
    
    # Add pathway features
    combined = add_pathway_features(data)
    print(f"\nCombined shape: {combined.shape}")
    print(f"Columns: {combined.columns.tolist()}")
