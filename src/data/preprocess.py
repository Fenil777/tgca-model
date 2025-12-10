"""Preprocessing utilities for TCGA data."""

from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..config import (
    PROCESSED_DATA_DIR,
    SURVIVAL_COLUMNS,
    TRAIN_TEST_SPLIT_CONFIG,
    RANDOM_SEED,
)
from ..utils.logging_utils import get_logger
from ..utils.io_utils import save_parquet, save_pickle

logger = get_logger(__name__)


def clean_clinical_features(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Clean and encode clinical features.
    
    Args:
        df: DataFrame with clinical features
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        
    Returns:
        DataFrame with cleaned features
    """
    logger.info("Cleaning clinical features")
    df = df.copy()
    
    # Identify categorical and numerical columns if not provided
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Remove patient_id and survival columns
        categorical_cols = [c for c in categorical_cols 
                          if c not in ['patient_id'] + SURVIVAL_COLUMNS]
    
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Remove survival columns
        numerical_cols = [c for c in numerical_cols if c not in SURVIVAL_COLUMNS]
    
    logger.info(f"Categorical columns: {categorical_cols}")
    logger.info(f"Numerical columns: {numerical_cols}")
    
    # One-hot encode categorical variables
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dummy_na=False)
        logger.info(f"One-hot encoded {len(categorical_cols)} categorical columns")
    
    # Fill missing numerical values with median
    for col in numerical_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"Filled missing values in {col} with median: {median_val}")
    
    return df


def normalize_expression(
    df: pd.DataFrame,
    expression_cols: Optional[List[str]] = None,
    method: str = "standardize",
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize gene expression data.
    
    Args:
        df: DataFrame with expression data
        expression_cols: List of expression column names (genes)
        method: Normalization method ('standardize' or 'minmax')
        
    Returns:
        Tuple of (normalized_df, scaler)
    """
    logger.info(f"Normalizing expression data using {method}")
    df = df.copy()
    
    # Identify expression columns if not provided
    if expression_cols is None:
        # Assume all numeric columns except survival are expression
        exclude_cols = ['patient_id'] + SURVIVAL_COLUMNS
        expression_cols = [c for c in df.columns 
                          if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    logger.info(f"Normalizing {len(expression_cols)} expression columns")
    
    if method == "standardize":
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    # Fit and transform
    df[expression_cols] = scaler.fit_transform(df[expression_cols])
    
    logger.info("Normalization complete")
    return df, scaler


def split_train_test(
    df: pd.DataFrame,
    test_size: Optional[float] = None,
    stratify_col: Optional[str] = None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets.
    
    Args:
        df: DataFrame to split
        test_size: Proportion of data for test set
        stratify_col: Column to stratify by
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info("Splitting data into train and test sets")
    
    test_size = test_size or TRAIN_TEST_SPLIT_CONFIG["test_size"]
    random_state = random_state or TRAIN_TEST_SPLIT_CONFIG["random_state"]
    stratify_col = stratify_col or TRAIN_TEST_SPLIT_CONFIG.get("stratify_by")
    
    stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")
    
    return train_df, test_df


def preprocess_pipeline(
    clinical_path: Path,
    expression_path: Path,
    output_dir: Optional[Path] = None,
    save_intermediate: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run complete preprocessing pipeline.
    
    Args:
        clinical_path: Path to clinical data
        expression_path: Path to expression data
        output_dir: Directory to save processed data
        save_intermediate: Whether to save intermediate files
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from .load_tcga import load_tcga_data, validate_survival_data
    
    logger.info("Starting preprocessing pipeline")
    
    output_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    merged_df, _ = load_tcga_data(clinical_path, expression_path, merge=True)
    
    # Validate survival data
    merged_df = validate_survival_data(merged_df)
    
    # Clean clinical features
    merged_df = clean_clinical_features(merged_df)
    
    # Normalize expression (identify gene columns automatically)
    merged_df, scaler = normalize_expression(merged_df)
    
    # Save scaler
    if save_intermediate:
        scaler_path = output_dir / "expression_scaler.pkl"
        save_pickle(scaler, scaler_path)
        logger.info(f"Saved expression scaler to {scaler_path}")
    
    # Split train/test
    train_df, test_df = split_train_test(merged_df)
    
    # Save processed data
    if save_intermediate:
        train_path = output_dir / "train.parquet"
        test_path = output_dir / "test.parquet"
        combined_path = output_dir / "combined.parquet"
        
        save_parquet(train_df, train_path)
        save_parquet(test_df, test_path)
        save_parquet(merged_df, combined_path)
        
        logger.info(f"Saved train data to {train_path}")
        logger.info(f"Saved test data to {test_path}")
        logger.info(f"Saved combined data to {combined_path}")
    
    logger.info("Preprocessing pipeline complete")
    return train_df, test_df


if __name__ == "__main__":
    from ..config import RAW_DATA_DIR
    
    # Example usage
    clinical_path = RAW_DATA_DIR / "BRCA_clinical.csv"
    expression_path = RAW_DATA_DIR / "BRCA_expression.csv"
    
    train_df, test_df = preprocess_pipeline(clinical_path, expression_path)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
