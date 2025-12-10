"""I/O utilities for TCGA survival prediction pipeline."""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)


def load_csv(file_path: Path, **kwargs) -> pd.DataFrame:
    """
    Load CSV file into pandas DataFrame.
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments to pass to pd.read_csv
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading CSV from {file_path}")
    df = pd.read_csv(file_path, **kwargs)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def save_csv(df: pd.DataFrame, file_path: Path, **kwargs) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        **kwargs: Additional arguments to pass to df.to_csv
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving CSV to {file_path}")
    df.to_csv(file_path, index=False, **kwargs)
    logger.info(f"Saved {len(df)} rows and {len(df.columns)} columns")


def load_parquet(file_path: Path, **kwargs) -> pd.DataFrame:
    """
    Load Parquet file into pandas DataFrame.
    
    Args:
        file_path: Path to Parquet file
        **kwargs: Additional arguments to pass to pd.read_parquet
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading Parquet from {file_path}")
    df = pd.read_parquet(file_path, **kwargs)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def save_parquet(df: pd.DataFrame, file_path: Path, **kwargs) -> None:
    """
    Save DataFrame to Parquet file.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        **kwargs: Additional arguments to pass to df.to_parquet
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving Parquet to {file_path}")
    df.to_parquet(file_path, index=False, **kwargs)
    logger.info(f"Saved {len(df)} rows and {len(df.columns)} columns")


def load_pickle(file_path: Path) -> Any:
    """
    Load object from pickle file.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Unpickled object
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading pickle from {file_path}")
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    logger.info("Pickle loaded successfully")
    return obj


def save_pickle(obj: Any, file_path: Path) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to pickle
        file_path: Output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving pickle to {file_path}")
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Pickle saved successfully")


def ensure_dir(dir_path: Path) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        dir_path: Directory path
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
