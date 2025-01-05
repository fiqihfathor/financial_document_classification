import mlflow
import os

def log_mlflow_params(config, exclude_params=None):
    """Log dynamic hyperparameters from a config file to MLflow"""
    exclude_params = exclude_params or []
    for key, value in config.items():
        if key not in exclude_params:
            mlflow.log_param(key, value)
            
def log_mlflow_metrics(epoch, metrics, exclude_metrics=None):
    """Log metrics dynamically to MLflow"""
    exclude_metrics = exclude_metrics or []
    for key, value in metrics.items():
        if key not in exclude_metrics:
            mlflow.log_metric(f"{key}_epoch_{epoch}", value)
    
def log_mlflow_artifacts(artifact_path, artifact_location=None):
    """Log artifacts to MLflow with optional dynamic artifact location"""
    artifact_location = artifact_location or "artifacts"
    if not os.path.exists(artifact_location):
        os.makedirs(artifact_location)
    mlflow.log_artifacts(artifact_path, artifact_location)

