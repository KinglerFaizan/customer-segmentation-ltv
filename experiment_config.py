"""
mlflow_tracking/experiment_config.py
--------------------------------------
MLflow experiment setup and logging helpers.
"""

import mlflow
import os


EXPERIMENTS = {
    'ltv': 'LTV_Modeling',
    'churn': 'Churn_Modeling',
    'segmentation': 'Customer_Segmentation',
}

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', './mlruns')


def setup_mlflow(experiment_key: str = 'ltv'):
    """Initialize MLflow tracking for a given experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_name = EXPERIMENTS.get(experiment_key, 'Default')
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Active experiment: {experiment_name}")


def log_dataset_info(df, name: str = 'transactions'):
    """Log basic dataset statistics to active MLflow run."""
    mlflow.log_param(f'{name}_rows', len(df))
    mlflow.log_param(f'{name}_cols', len(df.columns))
    if 'CustomerID' in df.columns:
        mlflow.log_param('unique_customers', df['CustomerID'].nunique())
