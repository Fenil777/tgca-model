# TCGA Multimodal Survival Prediction

A comprehensive machine learning pipeline for predicting cancer patient survival using multimodal data from The Cancer Genome Atlas (TCGA), with a focus on Breast Cancer (BRCA).

## Features

- **Multimodal Data Integration**: Combines clinical, genomic, and pathway-level features
- **Multiple Survival Models**: 
  - Cox Proportional Hazards (CoxPH)
  - Random Survival Forest (RSF)
  - Deep Learning Survival Model (DeepSurv)
- **Feature Engineering**: Gene filtering and pathway aggregation
- **Interpretability**: SHAP values for model explanation
- **Interactive App**: Streamlit-based web interface for predictions
- **Comprehensive Notebooks**: Step-by-step analysis and modeling

## Project Structure

```
tgca-model/
├── src/
│   ├── config.py                    # Configuration settings
│   ├── utils/
│   │   ├── logging_utils.py         # Logging utilities
│   │   └── io_utils.py              # I/O utilities
│   ├── data/
│   │   ├── download_tcga.py         # TCGA data download
│   │   ├── load_tcga.py             # Data loading
│   │   └── preprocess.py            # Preprocessing pipeline
│   ├── features/
│   │   ├── gene_filtering.py        # Gene selection
│   │   └── pathway_aggregation.py   # Pathway scoring
│   ├── models/
│   │   ├── cox_model.py             # Cox PH model
│   │   ├── rsf_model.py             # Random Survival Forest
│   │   └── deepsurv_model.py        # DeepSurv neural network
│   └── evaluation/
│       ├── metrics.py               # Evaluation metrics
│       ├── survival_plots.py        # Visualization
│       └── shap_utils.py            # SHAP explainability
├── app/
│   └── streamlit_app.py             # Interactive web app
├── notebooks/
│   ├── 01_exploration.ipynb         # Data exploration
│   ├── 02_feature_engineering.ipynb # Feature engineering
│   ├── 03_modeling_classical.ipynb  # Classical models
│   ├── 04_modeling_deepsurv.ipynb   # Deep learning
│   └── 05_interpretability.ipynb    # Model interpretation
├── tests/
│   ├── test_data_pipeline.py        # Data pipeline tests
│   ├── test_models.py               # Model tests
│   └── test_evaluation.py           # Evaluation tests
├── data/
│   └── processed/                   # Processed data files
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Fenil777/tgca-model.git
cd tgca-model
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### 1. Data Preparation

Download and preprocess TCGA data:

```python
from src.data.download_tcga import download_tcga_data
from src.data.preprocess import preprocess_pipeline

# Download data (requires TCGA credentials)
download_tcga_data(cancer_type='BRCA', output_dir='data/raw/')

# Preprocess
preprocess_pipeline(
    clinical_path='data/raw/clinical.csv',
    expression_path='data/raw/expression.csv',
    output_dir='data/processed/'
)
```

### 2. Feature Engineering

```python
from src.features.gene_filtering import select_variable_genes
from src.features.pathway_aggregation import compute_pathway_scores

# Select variable genes
selected_genes = select_variable_genes(expression_data, top_n=1000)

# Compute pathway scores
pathway_scores = compute_pathway_scores(expression_data)
```

### 3. Model Training

```python
from src.models.cox_model import train_cox_model
from src.models.rsf_model import train_rsf_model
from src.models.deepsurv_model import train_deepsurv_model

# Train Cox model
cox_model = train_cox_model(X_train, y_train)

# Train RSF model
rsf_model = train_rsf_model(X_train, y_train)

# Train DeepSurv model
deepsurv_model = train_deepsurv_model(X_train, y_train)
```

### 4. Evaluation

```python
from src.evaluation.metrics import compute_concordance_index
from src.evaluation.survival_plots import plot_kaplan_meier

# Compute C-index
c_index = compute_concordance_index(model, X_test, y_test)

# Plot Kaplan-Meier curves
plot_kaplan_meier(y_test, predictions, groups=['high_risk', 'low_risk'])
```

### 5. Interactive App

Launch the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Navigate to `http://localhost:8501` in your browser.

### 6. Notebooks

Explore the Jupyter notebooks for detailed analysis:

```bash
jupyter notebook notebooks/
```

## Running Tests

Run all tests:

```bash
pytest tests/
```

Run specific test files:

```bash
pytest tests/test_data_pipeline.py
pytest tests/test_models.py
pytest tests/test_evaluation.py
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Models

### Cox Proportional Hazards (CoxPH)
A semi-parametric regression model that estimates the hazard function. Best for linear relationships and interpretability.

### Random Survival Forest (RSF)
An ensemble method that handles non-linear relationships and feature interactions. Robust to outliers.

### DeepSurv
A deep neural network that learns complex non-linear patterns in survival data. Requires more data but can capture intricate relationships.

## Interpretability

We use SHAP (SHapley Additive exPlanations) to explain model predictions:

```python
from src.evaluation.shap_utils import compute_shap_values, plot_shap_summary

# Compute SHAP values
shap_values = compute_shap_values(model, X_test)

# Plot summary
plot_shap_summary(shap_values, X_test)
```

## Configuration

Edit `src/config.py` to customize:
- Cancer type (default: BRCA)
- Model hyperparameters
- Feature selection thresholds
- Paths and directories

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
@software{tcga_multimodal_survival,
  title = {TCGA Multimodal Survival Prediction},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/Fenil777/tgca-model}
}
```

## Acknowledgments

- The Cancer Genome Atlas (TCGA) for providing the data
- lifelines, scikit-survival, and PyTorch communities
- SHAP library for interpretability tools

## Contact

For questions or issues, please open an issue on GitHub.