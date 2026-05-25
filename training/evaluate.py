import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from training import feature_engineering
from training import logs


PARENT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PARENT_DIR / "models"
METRICS_DIR = PARENT_DIR / "evaluation_metrics"
DEFAULT_DATASET_MODE = "sample_collider"
EVALUATION_REPORTS_DIR = PARENT_DIR / "evaluation_reports"

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
        "hist_gradient_boosting": MODEL_DIR / f"hist_gradient_boosting_{dataset_mode}.joblib",
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
    hist_gradient_boosting_model = joblib.load(model_paths["hist_gradient_boosting"])

    return logistic_model, random_forest_model, hist_gradient_boosting_model, model_paths


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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate classification metrics and detailed evaluation artifacts for one model."""
    print(f"Generating predictions for model: {model_name}")
    prediction = model.predict(predictors)

    roc_auc = None
    if hasattr(model, "predict_proba"):
        model_classes = list(model.classes_)
        positive_class = 1 if 1 in model_classes else "signal"
        positive_class_index = model_classes.index(positive_class)
        prediction_probability = model.predict_proba(predictors)[:, positive_class_index]
        roc_auc = roc_auc_score(target, prediction_probability)

    recall = recall_score(target, prediction, zero_division=0)
    precision = precision_score(target, prediction, zero_division=0)
    accuracy = accuracy_score(target, prediction)
    f1 = f1_score(target, prediction, zero_division=0)

    labels = list(model.classes_)
    confusion_matrix_values = confusion_matrix(
        target,
        prediction,
        labels=labels,
    )

    classification_report_dict = classification_report(
        target,
        prediction,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    performance_df = pd.DataFrame(
        {
            "model_name": model_name,
            "recall_score": recall,
            "precision_score": precision,
            "accuracy_score": accuracy,
            "f1_score": f1,
            "roc_auc_score": roc_auc,
        },
        index=[0],
    )

    detailed_artifacts = {
        "model_name": model_name,
        "labels": labels,
        "confusion_matrix": confusion_matrix_values,
        "classification_report": classification_report_dict,
        "model": model,
        "predictors": predictors,
        "target": target,
    }

    return performance_df, detailed_artifacts

def write_confusion_matrix_artifact(
    artifact: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Write one model confusion matrix to CSV."""

    labels = artifact["labels"]
    matrix = artifact["confusion_matrix"]
    model_name = artifact["model_name"]

    confusion_matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )

    output_path = output_dir / f"{model_name}_confusion_matrix.csv"
    confusion_matrix_df.to_csv(output_path)
    return output_path

def write_classification_report_artifact(
    artifact: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Write one model classification report to CSV."""

    model_name = artifact["model_name"]
    report_df = pd.DataFrame(artifact["classification_report"]).transpose()

    output_path = output_dir / f"{model_name}_classification_report.csv"
    report_df.to_csv(output_path)
    return output_path

def write_builtin_feature_importance_artifact(
    artifact: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    """Write built-in feature importance when the model exposes feature_importances_."""
    model = artifact["model"]
    predictors = artifact["predictors"]
    model_name = artifact["model_name"]

    if not hasattr(model, "feature_importances_"):
        return None

    feature_importance_df = pd.DataFrame(
        {
            "feature": predictors.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    output_path = output_dir / f"{model_name}_feature_importance.csv"
    feature_importance_df.to_csv(output_path, index=False)
    return output_path


def write_permutation_importance_artifact(
    artifact: dict[str, Any],
    output_dir: Path,
    n_repeats: int = 10,
) -> Path:
    """Write permutation importance for any fitted classifier."""
    model = artifact["model"]
    predictors = artifact["predictors"]
    target = artifact["target"]
    model_name = artifact["model_name"]

    permutation_result = permutation_importance(
        model,
        predictors,
        target,
        n_repeats=n_repeats,
        random_state=42,
        scoring="f1",
    )

    permutation_importance_df = pd.DataFrame(
        {
            "feature": predictors.columns,
            "importance_mean": permutation_result.importances_mean,
            "importance_std": permutation_result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    output_path = output_dir / f"{model_name}_permutation_importance.csv"
    permutation_importance_df.to_csv(output_path, index=False)
    return output_path


# Helper: build permutation importance summary and evaluation summary artifact
def build_top_permutation_features_summary(
    artifact: dict[str, Any],
    output_dir: Path,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Read a model permutation importance artifact and return the top features."""
    model_name = artifact["model_name"]
    permutation_path = output_dir / f"{model_name}_permutation_importance.csv"

    if not permutation_path.exists():
        return []

    permutation_df = pd.read_csv(permutation_path)
    top_features = permutation_df.head(top_n).to_dict(orient="records")
    return top_features


def write_evaluation_summary_artifact(
    dataset_mode: str,
    timestamp: str,
    comparison_df: pd.DataFrame,
    artifacts: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write a human-readable markdown summary for one evaluation run."""
    best_accuracy_row = comparison_df.sort_values(
        "accuracy_score",
        ascending=False,
    ).iloc[0]
    best_f1_row = comparison_df.sort_values("f1_score", ascending=False).iloc[0]
    best_roc_auc_row = comparison_df.sort_values(
        "roc_auc_score",
        ascending=False,
    ).iloc[0]

    summary_lines = [
        f"# Evaluation Summary: {dataset_mode}",
        "",
        f"Timestamp: `{timestamp}`",
        "",
        "## Model comparison",
        "",
        comparison_df.to_markdown(index=False),
        "",
        "## Best models",
        "",
        f"- Best accuracy: `{best_accuracy_row['model_name']}` "
        f"({best_accuracy_row['accuracy_score']:.4f})",
        f"- Best F1: `{best_f1_row['model_name']}` "
        f"({best_f1_row['f1_score']:.4f})",
        f"- Best ROC-AUC: `{best_roc_auc_row['model_name']}` "
        f"({best_roc_auc_row['roc_auc_score']:.4f})",
        "",
        "## Top permutation-importance features",
        "",
    ]

    for artifact in artifacts:
        model_name = artifact["model_name"]
        top_features = build_top_permutation_features_summary(
            artifact=artifact,
            output_dir=output_dir,
        )

        summary_lines.extend([f"### {model_name}", ""])
        if not top_features:
            summary_lines.extend(["No permutation-importance artifact found.", ""])
            continue

        for feature in top_features:
            summary_lines.append(
                f"- `{feature['feature']}`: "
                f"mean={feature['importance_mean']:.6f}, "
                f"std={feature['importance_std']:.6f}"
            )
        summary_lines.append("")

    summary_lines.extend(
        [
            "## Interpretation notes",
            "",
            "- Accuracy, precision, recall, and F1 evaluate the model at its default classification threshold.",
            "- ROC-AUC evaluates how well the model ranks signal-like events above background-like events across thresholds.",
            "- Permutation importance estimates how much model performance drops when each feature is shuffled.",
            "- Features with high mean importance and relatively low standard deviation are stronger, more stable drivers of model performance.",
            "",
        ]
    )

    output_path = output_dir / "evaluation_summary.md"
    output_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return output_path


def write_detailed_evaluation_artifacts(
    dataset_mode: str,
    timestamp: str,
    artifacts: list[dict[str, Any]],
) -> list[Path]:
    """Write confusion matrix and classification report artifacts for all models."""
    output_dir = EVALUATION_REPORTS_DIR / dataset_mode / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []

    for artifact in artifacts:
        output_paths.append(
            write_confusion_matrix_artifact(
                artifact=artifact,
                output_dir=output_dir,
            )
        )

        output_paths.append(
            write_classification_report_artifact(
                artifact=artifact,
                output_dir=output_dir,
            )
        )
        builtin_feature_importance_path = write_builtin_feature_importance_artifact(
            artifact=artifact,
            output_dir=output_dir,
        )
        if builtin_feature_importance_path is not None:
            output_paths.append(builtin_feature_importance_path)

        output_paths.append(
            write_permutation_importance_artifact(
                artifact=artifact,
                output_dir=output_dir,
            )
        )
    return output_paths

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
            "models_evaluated": "logistic_regression, random_forest, hist_gradient_boosting",
            "logistic_model_path": str(model_paths["logistic_regression"]),
            "random_forest_model_path": str(model_paths["random_forest"]),
            "hist_gradient_boosting_model_path": str(model_paths["hist_gradient_boosting"]),
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
    hist_gradient_boosting_model_metrics: pd.DataFrame,
) -> None:
    """Log evaluation metrics to MLflow."""
    logs.log_training_metrics(
        {
            "logistic_recall": logistic_model_metrics.loc[0, "recall_score"],
            "logistic_precision": logistic_model_metrics.loc[0, "precision_score"],
            "logistic_accuracy": logistic_model_metrics.loc[0, "accuracy_score"],
            "logistic_f1": logistic_model_metrics.loc[0, "f1_score"],
            "logistic_roc_auc": logistic_model_metrics.loc[0, "roc_auc_score"],
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
            "random_forest_roc_auc": random_forest_model_metrics.loc[
                0,
                "roc_auc_score",
            ],
            "hist_gradient_boosting_recall": hist_gradient_boosting_model_metrics.loc[
                0,
                "recall_score",
            ],
            "hist_gradient_boosting_precision": hist_gradient_boosting_model_metrics.loc[
                0,
                "precision_score",
            ],
            "hist_gradient_boosting_accuracy": hist_gradient_boosting_model_metrics.loc[
                0,
                "accuracy_score",
            ],
            "hist_gradient_boosting_f1": hist_gradient_boosting_model_metrics.loc[
                0,
                "f1_score",
            ],
            "hist_gradient_boosting_roc_auc": hist_gradient_boosting_model_metrics.loc[
                0,
                "roc_auc_score",
            ],
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
    (
        logistic_model,
        random_forest_model,
        hist_gradient_boosting_model,
        model_paths,
    ) = load_saved_models(dataset_mode=dataset_mode)

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

        logistic_model_metrics, logistic_artifact = compare_performance(
            model=logistic_model,
            predictors=x_test,
            target=y_test,
            model_name=f"logistic_regression_{dataset_mode}",
        )

        random_forest_model_metrics, random_forest_artifact = compare_performance(
            model=random_forest_model,
            predictors=x_test,
            target=y_test,
            model_name=f"random_forest_{dataset_mode}",
        )

        hist_gradient_boosting_model_metrics, hist_gradient_boosting_artifact = compare_performance(
            model=hist_gradient_boosting_model,
            predictors=x_test,
            target=y_test,
            model_name=f"hist_gradient_boosting_{dataset_mode}",
        )

        log_evaluation_metrics(
            logistic_model_metrics=logistic_model_metrics,
            random_forest_model_metrics=random_forest_model_metrics,
            hist_gradient_boosting_model_metrics=hist_gradient_boosting_model_metrics,
        )

        comparison_df = pd.concat(
            [
                logistic_model_metrics,
                random_forest_model_metrics,
                hist_gradient_boosting_model_metrics,
            ],
            ignore_index=True,
        )

        comparison_df.to_csv(metrics_output_path, index=False)
        logs.log_file_artifact(
            file_path=str(metrics_output_path),
            artifact_path="evaluation_metrics",
        )

        detailed_artifact_paths = write_detailed_evaluation_artifacts(
            dataset_mode=dataset_mode,
            timestamp=timestamp,
            artifacts=[
                logistic_artifact,
                random_forest_artifact,
                hist_gradient_boosting_artifact,
            ],
        )

        detailed_output_dir = EVALUATION_REPORTS_DIR / dataset_mode / timestamp
        evaluation_summary_path = write_evaluation_summary_artifact(
            dataset_mode=dataset_mode,
            timestamp=timestamp,
            comparison_df=comparison_df,
            artifacts=[
                logistic_artifact,
                random_forest_artifact,
                hist_gradient_boosting_artifact,
            ],
            output_dir=detailed_output_dir,
        )
        detailed_artifact_paths.append(evaluation_summary_path)

        for artifact_path in detailed_artifact_paths:
            logs.log_file_artifact(
                file_path=str(artifact_path),
                artifact_path="evaluation_reports",
            )

        print(f"Evaluation metrics saved at: {metrics_output_path}")
        print(
            f"Detailed evaluation artifacts saved under: "
            f"{EVALUATION_REPORTS_DIR / dataset_mode / timestamp}"
        )
        print(f"Evaluation summary saved at: {evaluation_summary_path}")
        print(comparison_df)


if __name__ == "__main__":
    main()
