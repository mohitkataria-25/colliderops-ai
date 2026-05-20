from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_TRACKING_URI = MLFLOW_TRACKING_DIR.as_uri()
EXPERIMENT_NAME = "ColliderOpsAI"

def configure_mlflow(
        experiment_name:str = EXPERIMENT_NAME,
        tracking_uri: str = MLFLOW_TRACKING_URI,
):
    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    return experiment_id

def log_training_params(params:dict[str, Any]):

    for key, value in params.items():
        mlflow.log_param(key, value)

def log_training_metrics(metrics:dict[str, Any]):

    for key, value in metrics.items():
        if value is not None:
            mlflow.log_metric(key, float(value))

def log_model_artifact(model_path: str | Path)->None:

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    
    mlflow.log_artifact(str(model_path), artifact_path="models")

def log_file_artifact(file_path: str | Path, artifact_path: str = "artifacts")->None:

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File artifact not found: {file_path}")
    
    mlflow.log_artifact(str(file_path), artifact_path=artifact_path)

def start_training_run(run_name:str):

    experiment_id = configure_mlflow()
    return mlflow.start_run(
        run_name=run_name,
        experiment_id=experiment_id,
        )