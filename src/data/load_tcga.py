"""Load and validate TCGA data."""

from pathlib import Path
from typing import Tuple, Optional

import pandas as pd

from ..config import RAW_DATA_DIR, CLINICAL_FEATURES, SURVIVAL_COLUMNS
from ..utils.logging_utils import get_logger
from ..utils.io_utils import load_csv

logger = get_logger(__name__)


def load_clinical_data(
    file_path: Optional[Path] = None,
    required_columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load clinical data from CSV file.
    
    Args:
        file_path: Path to clinical data file
        required_columns: List of required column names
        
    Returns:
        DataFrame with clinical data
        
    Raises:
        ValueError: If required columns are missing
    """
    if file_path is None:
        file_path = RAW_DATA_DIR / "BRCA_clinical.csv"
    
    df = load_csv(file_path)
    logger.info(f"Loaded clinical data with shape {df.shape}")
    
    # Validate required columns
    required_columns = required_columns or (SURVIVAL_COLUMNS + ['patient_id'])
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Basic validation
    if 'patient_id' in df.columns:
        logger.info(f"Found {df['patient_id'].nunique()} unique patients")
    
    return df


def load_expression_data(
    file_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load gene expression data from CSV file.
    
    Args:
        file_path: Path to expression data file
        
    Returns:
        DataFrame with expression data (samples x genes)
    """
    if file_path is None:
        file_path = RAW_DATA_DIR / "BRCA_expression.csv"
    
    df = load_csv(file_path)
    logger.info(f"Loaded expression data with shape {df.shape}")
    
    if 'patient_id' in df.columns:
        logger.info(f"Found {df['patient_id'].nunique()} unique samples")
        n_genes = len(df.columns) - 1
        logger.info(f"Found {n_genes} genes")
    
    return df


def merge_clinical_expression(
    clinical_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    patient_id_col: str = "patient_id",
) -> pd.DataFrame:
    """
    Merge clinical and expression data.
    
    Args:
        clinical_df: Clinical data DataFrame
        expression_df: Expression data DataFrame
        patient_id_col: Name of patient ID column
        
    Returns:
        Merged DataFrame
        
    Raises:
        ValueError: If no common patients found
    """
    logger.info("Merging clinical and expression data")
    
    # Check for common patients
    clinical_ids = set(clinical_df[patient_id_col])
    expression_ids = set(expression_df[patient_id_col])
    common_ids = clinical_ids & expression_ids
    
    if not common_ids:
        raise ValueError("No common patients found between clinical and expression data")
    
    logger.info(f"Found {len(common_ids)} common patients")
    logger.info(f"  Clinical-only: {len(clinical_ids - expression_ids)}")
    logger.info(f"  Expression-only: {len(expression_ids - clinical_ids)}")
    
    # Merge
    merged_df = clinical_df.merge(
        expression_df,
        on=patient_id_col,
        how="inner"
    )
    
    logger.info(f"Merged data shape: {merged_df.shape}")
    return merged_df


def load_tcga_data(
    clinical_path: Optional[Path] = None,
    expression_path: Optional[Path] = None,
    merge: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load TCGA clinical and expression data.
    
    Args:
        clinical_path: Path to clinical data
        expression_path: Path to expression data
        merge: Whether to merge clinical and expression data
        
    Returns:
        Tuple of (clinical_df, expression_df) or (merged_df, None) if merge=True
    """
    logger.info("Loading TCGA data")
    
    clinical_df = load_clinical_data(clinical_path)
    expression_df = load_expression_data(expression_path)
    
    if merge:
        merged_df = merge_clinical_expression(clinical_df, expression_df)
        return merged_df, None
    else:
        return clinical_df, expression_df


def validate_survival_data(
    df: pd.DataFrame,
    time_col: str = "OS_time",
    event_col: str = "OS_status",
) -> pd.DataFrame:
    """
    Validate and clean survival data.
    
    Args:
        df: DataFrame with survival data
        time_col: Name of survival time column
        event_col: Name of event status column
        
    Returns:
        Validated DataFrame
        
    Raises:
        ValueError: If survival columns are invalid
    """
    logger.info("Validating survival data")
    
    if time_col not in df.columns or event_col not in df.columns:
        raise ValueError(f"Missing survival columns: {time_col}, {event_col}")
    
    # Check for missing values
    missing_time = df[time_col].isna().sum()
    missing_event = df[event_col].isna().sum()
    
    if missing_time > 0 or missing_event > 0:
        logger.warning(f"Missing values - time: {missing_time}, event: {missing_event}")
        df = df.dropna(subset=[time_col, event_col])
        logger.info(f"Dropped rows with missing survival data. New shape: {df.shape}")
    
    # Check for negative times
    negative_times = (df[time_col] < 0).sum()
    if negative_times > 0:
        logger.warning(f"Found {negative_times} negative survival times. Removing...")
        df = df[df[time_col] >= 0]
    
    # Event status should be 0 or 1
    valid_events = df[event_col].isin([0, 1]).sum()
    if valid_events != len(df):
        logger.warning(f"Invalid event status values found. Converting to binary...")
        df[event_col] = (df[event_col] > 0).astype(int)
    
    # Summary statistics
    logger.info(f"Survival data summary:")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Events: {df[event_col].sum()} ({df[event_col].mean()*100:.1f}%)")
    logger.info(f"  Censored: {(1-df[event_col]).sum()} ({(1-df[event_col].mean())*100:.1f}%)")
    logger.info(f"  Median survival time: {df[time_col].median():.1f}")
    
    return df


if __name__ == "__main__":
    # Example usage
    merged_df, _ = load_tcga_data()
    validated_df = validate_survival_data(merged_df)
    print(validated_df.head())
