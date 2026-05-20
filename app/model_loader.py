from pathlib import Path
import json

import joblib


PARENT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PARENT_DIR / "models"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"


def get_model_path(model_to_load: str) -> Path:
    """Return the full path for a model artifact."""
    return MODEL_DIR / model_to_load


def load_model(model_to_load: str):
    """Load a saved model artifact from the models directory."""
    model_path = get_model_path(model_to_load=model_to_load)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)


def load_model_metadata_file() -> dict:
    """Load model metadata from models/model_metadata.json."""
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(f"Model metadata file not found: {MODEL_METADATA_PATH}")

    with MODEL_METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_active_model_name() -> str:
    """Return the active model artifact name from metadata."""
    metadata = load_model_metadata_file()

    active_model = metadata.get("active_model")

    if not active_model:
        raise ValueError("active_model is missing from model_metadata.json")

    return active_model


def get_model_metadata(model_to_load: str | None = None) -> dict:
    """Return metadata for the selected model."""
    metadata = load_model_metadata_file()

    active_model = metadata.get("active_model")
    selected_model = model_to_load or active_model

    if not selected_model:
        raise ValueError("No model selected and no active_model found in metadata.")

    model_path = get_model_path(model_to_load=selected_model)
    selected_model_name = selected_model.replace(".joblib", "")

    return {
        "model_name": metadata.get("model_name", selected_model_name),
        "model_version": metadata.get("model_version", "unknown"),
        "model_type": metadata.get("model_type", "unknown"),
        "model_stage": metadata.get("model_stage", "unknown"),
        "model_file": selected_model,
        "model_path": str(model_path),
        "active_model": active_model,
        "training_data_path": metadata.get("training_data_path"),
        "feature_columns": metadata.get("feature_columns", []),
        "confidence_threshold": metadata.get("confidence_threshold"),
        "latest_metrics": metadata.get("latest_metrics", {}),
        "mlflow": metadata.get("mlflow", {}),
    }