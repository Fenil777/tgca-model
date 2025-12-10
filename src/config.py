"""Configuration settings for TCGA survival prediction pipeline."""

import os
from pathlib import Path
from typing import Dict, List, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Cancer type configuration
DEFAULT_CANCER_TYPE: str = "BRCA"
SUPPORTED_CANCER_TYPES: List[str] = [
    "BRCA", "LUAD", "LUSC", "COAD", "READ", "KIRC", "KIRP", "OV"
]

# Data configuration
CLINICAL_FEATURES: List[str] = [
    "age_at_diagnosis",
    "gender",
    "race",
    "ethnicity",
    "tumor_stage",
    "tumor_grade",
    "ajcc_pathologic_stage",
]

SURVIVAL_COLUMNS: List[str] = ["OS_time", "OS_status"]

# Feature engineering configuration
GENE_FILTERING_CONFIG: Dict[str, Any] = {
    "variance_threshold": 0.1,
    "top_n_genes": 1000,
    "min_expression": 1.0,
}

# Pathway configuration (example pathways)
PATHWAY_GENES: Dict[str, List[str]] = {
    "PI3K_AKT": ["PIK3CA", "AKT1", "AKT2", "AKT3", "PTEN", "MTOR"],
    "RAS_MAPK": ["KRAS", "NRAS", "HRAS", "BRAF", "MAP2K1", "MAPK1", "MAPK3"],
    "P53": ["TP53", "MDM2", "MDM4", "CDKN2A", "ATM", "CHEK2"],
    "WNT": ["CTNNB1", "APC", "AXIN1", "AXIN2", "TCF7L2", "LEF1"],
    "NOTCH": ["NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "JAG1", "DLL1"],
    "TGF_BETA": ["TGFB1", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SMAD4"],
    "CELL_CYCLE": ["CDK4", "CDK6", "CCND1", "CCND2", "CCND3", "RB1"],
    "APOPTOSIS": ["BCL2", "BAX", "BAK1", "CASP3", "CASP8", "CASP9"],
}

# Model configuration
COX_MODEL_CONFIG: Dict[str, Any] = {
    "penalizer": 0.01,
    "l1_ratio": 0.5,
}

RSF_MODEL_CONFIG: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": 42,
}

DEEPSURV_CONFIG: Dict[str, Any] = {
    "hidden_dims": [64, 32, 16],
    "activation": "relu",
    "dropout": 0.3,
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "early_stopping_patience": 10,
}

# Training configuration
TRAIN_TEST_SPLIT_CONFIG: Dict[str, Any] = {
    "test_size": 0.2,
    "random_state": 42,
    "stratify_by": "OS_status",
}

# Evaluation configuration
SURVIVAL_TIME_POINTS: List[int] = [365, 730, 1095, 1825]  # 1, 2, 3, 5 years

# Logging configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Random seed for reproducibility
RANDOM_SEED: int = 42
