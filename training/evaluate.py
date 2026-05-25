
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from training import feature_engineering
from training import logs


PARENT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PARENT_DIR / "models"
METRICS_DIR = PARENT_DIR / "evaluation_metrics"
DEFAULT_DATASET_MODE = "sample_collider"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-mode",
        default=DEFAULT_DATASET_MODE,
        choices=["sample_collider", "real_cern", "curated_higgs"],
        help="Choose which curated dataset to evaluate on.",
    )

    return parser.parse_args()


def resolve_evaluation_data_path(dataset_mode: str) -> Path:
    """Return the curated evaluation data path for the selected dataset mode."""
    if dataset_mode == "sample_collider":
        return feature_engineering.SAMPLE_CURATED_DATA_DIR

    if dataset_mode == "real_cern":
        return feature_engineering.REAL_CERN_CURATED_DATA_PATH

    if dataset_mode == "curated_higgs":
        return feature_engineering.CURATED_HIGGS_CURATED_DATA_PATH

    raise ValueError(
        f"Unsupported dataset_mode={dataset_mode}. "
        "Supported values: sample_collider, real_cern, curated_higgs"
    )


def resolve_model_paths(dataset_mode: str) -> dict[str, Path]:
    """Return model paths for the selected dataset mode."""
    return {
        "logistic_regression": MODEL_DIR / f"logistic_regression_{dataset_mode}.joblib",
        "random_forest": MODEL_DIR / f"random_forest_{dataset_mode}.joblib",
    }


def load_saved_models(dataset_mode: str):
    """Load trained models for the selected dataset mode."""
    model_paths = resolve_model_paths(dataset_mode=dataset_mode)

    missing_models = [
        str(model_path)
        for model_path in model_paths.values()
        if not model_path.exists()
    ]

    if missing_models:
        raise FileNotFoundError(
            "Missing trained model artifacts. Train models before evaluation. "
            f"Missing: {missing_models}"
        )

    logistic_model = joblib.load(model_paths["logistic_regression"])
    random_forest_model = joblib.load(model_paths["random_forest"])

    return logistic_model, random_forest_model, model_paths


def load_test_data(
    dataset_mode: str,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load and split test data for the selected dataset mode."""
    df = feature_engineering.read_curated_training_data(dataset_mode=dataset_mode)

    validation_summary = feature_engineering.validate_model_ready_dataset(
        df=df,
        dataset_mode=dataset_mode,
        require_two_classes=True,
    )

    _, x_test, _, y_test = feature_engineering.split_features_and_label(
        df=df,
        dataset_mode=dataset_mode,
    )

    return x_test, y_test, validation_summary


def compare_performance(
    model,
    predictors: pd.DataFrame,
    target: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    """Generate classification metrics for one model."""
    print(f"Generating predictions for model: {model_name}")
    prediction = model.predict(predictors)

    recall = recall_score(target, prediction, zero_division=0)
    precision = precision_score(target, prediction, zero_division=0)
    accuracy = accuracy_score(target, prediction)
    f1 = f1_score(target, prediction, zero_division=0)

    performance_df = pd.DataFrame(
        {
            "model_name": model_name,
            "recall_score": recall,
            "precision_score": precision,
            "accuracy_score": accuracy,
            "f1_score": f1,
        },
        index=[0],
    )

    return performance_df


def log_evaluation_metadata(
    dataset_mode: str,
    evaluation_data_path: Path,
    feature_columns: list[str],
    model_paths: dict[str, Path],
    validation_summary: dict[str, Any],
    x_test: pd.DataFrame,
    metrics_output_path: Path,
) -> None:
    """Log evaluation parameters and dataset metadata to MLflow."""
    logs.log_training_params(
        {
            "dataset_mode": dataset_mode,
            "evaluation_data_path": str(evaluation_data_path),
            "feature_columns": ",".join(feature_columns),
            "models_evaluated": "logistic_regression, random_forest",
            "logistic_model_path": str(model_paths["logistic_regression"]),
            "random_forest_model_path": str(model_paths["random_forest"]),
            "test_rows": len(x_test),
            "metrics_output_path": str(metrics_output_path),
            "label_counts": str(validation_summary.get("label_counts")),
            "two_class_training_ready": validation_summary.get(
                "two_class_training_ready"
            ),
            "schema_valid": validation_summary.get("schema_valid"),
            "features_numeric": validation_summary.get("features_numeric"),
            "null_feature_count": validation_summary.get("null_feature_count"),
        }
    )


def log_evaluation_metrics(
    logistic_model_metrics: pd.DataFrame,
    random_forest_model_metrics: pd.DataFrame,
) -> None:
    """Log evaluation metrics to MLflow."""
    logs.log_training_metrics(
        {
            "logistic_recall": logistic_model_metrics.loc[0, "recall_score"],
            "logistic_precision": logistic_model_metrics.loc[0, "precision_score"],
            "logistic_accuracy": logistic_model_metrics.loc[0, "accuracy_score"],
            "logistic_f1": logistic_model_metrics.loc[0, "f1_score"],
            "random_forest_recall": random_forest_model_metrics.loc[0, "recall_score"],
            "random_forest_precision": random_forest_model_metrics.loc[
                0,
                "precision_score",
            ],
            "random_forest_accuracy": random_forest_model_metrics.loc[
                0,
                "accuracy_score",
            ],
            "random_forest_f1": random_forest_model_metrics.loc[0, "f1_score"],
        }
    )


def main() -> None:
    """Evaluate trained baseline models for the selected dataset mode."""
    args = parse_args()
    dataset_mode = args.dataset_mode

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_output_path = METRICS_DIR / f"model_comparison_{dataset_mode}_{timestamp}.csv"

    print(f"Loading saved models for dataset_mode={dataset_mode}...")
    logistic_model, random_forest_model, model_paths = load_saved_models(
        dataset_mode=dataset_mode,
    )

    print(f"Loading test data for dataset_mode={dataset_mode}...")
    x_test, y_test, validation_summary = load_test_data(dataset_mode=dataset_mode)

    feature_columns = feature_engineering.get_feature_columns(dataset_mode=dataset_mode)
    evaluation_data_path = resolve_evaluation_data_path(dataset_mode=dataset_mode)

    with logs.start_training_run(run_name=f"evaluate_baseline_models_{dataset_mode}"):
        log_evaluation_metadata(
            dataset_mode=dataset_mode,
            evaluation_data_path=evaluation_data_path,
            feature_columns=feature_columns,
            model_paths=model_paths,
            validation_summary=validation_summary,
            x_test=x_test,
            metrics_output_path=metrics_output_path,
        )

        logistic_model_metrics = compare_performance(
            model=logistic_model,
            predictors=x_test,
            target=y_test,
            model_name=f"logistic_regression_{dataset_mode}",
        )

        random_forest_model_metrics = compare_performance(
            model=random_forest_model,
            predictors=x_test,
            target=y_test,
            model_name=f"random_forest_{dataset_mode}",
        )

        log_evaluation_metrics(
            logistic_model_metrics=logistic_model_metrics,
            random_forest_model_metrics=random_forest_model_metrics,
        )

        comparison_df = pd.concat(
            [logistic_model_metrics, random_forest_model_metrics],
            ignore_index=True,
        )

        comparison_df.to_csv(metrics_output_path, index=False)
        logs.log_file_artifact(
            file_path=str(metrics_output_path),
            artifact_path="evaluation_metrics",
        )

        print(f"Evaluation metrics saved at: {metrics_output_path}")
        print(comparison_df)


if __name__ == "__main__":
    main()
