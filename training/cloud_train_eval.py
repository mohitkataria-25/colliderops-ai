"""Cloud training and evaluation entrypoint for ColliderOpsAI.

This script is intended to run inside an ECS/Fargate task. It reads the curated
HIGGS Parquet dataset from S3, trains baseline sklearn models, evaluates them,
and writes model/evaluation artifacts back to S3.

Expected environment variables:

COLLIDEROPS_S3_BUCKET=colliderops-ai-dev
COLLIDEROPS_S3_PREFIX=dev
COLLIDEROPS_DATASET_MODE=curated_higgs
COLLIDEROPS_RUN_ID=curated_higgs-YYYYMMDDTHHMMSSZ
COLLIDEROPS_TEST_SIZE=0.2
COLLIDEROPS_RANDOM_STATE=42

Expected input:

s3://<bucket>/<prefix>/curated/curated_higgs/training_dataset.parquet/

Expected outputs:

s3://<bucket>/<prefix>/models/<dataset_mode>/<run_id>/<model_name>.joblib
s3://<bucket>/<prefix>/evaluation/<dataset_mode>/<run_id>/model_comparison.csv
s3://<bucket>/<prefix>/evaluation/<dataset_mode>/<run_id>/evaluation_summary.md
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import joblib
import pandas as pd
from botocore.exceptions import ClientError
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SUPPORTED_DATASET_MODES = {"curated_higgs"}
DEFAULT_S3_PREFIX = "dev"
DEFAULT_DATASET_MODE = "curated_higgs"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
LABEL_COLUMN = "label"
POSITIVE_LABEL = "signal"
NEGATIVE_LABEL = "background"


class CloudTrainEvalError(ValueError):
    """Raised when cloud training/evaluation cannot complete safely."""


@dataclass(frozen=True)
class CloudTrainEvalConfig:
    """Runtime configuration for cloud training/evaluation."""

    s3_bucket: str
    s3_prefix: str
    dataset_mode: str
    run_id: str
    test_size: float
    random_state: int


def utc_timestamp() -> str:
    """Return a compact UTC timestamp for run IDs."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_prefix(s3_prefix: str | None) -> str:
    """Normalize an S3 prefix by removing leading/trailing slashes."""
    if not s3_prefix:
        return DEFAULT_S3_PREFIX

    return s3_prefix.strip().strip("/") or DEFAULT_S3_PREFIX


def build_s3_uri(bucket: str, key: str) -> str:
    """Build an S3 URI from bucket and key."""
    return f"s3://{bucket}/{key.strip('/')}"


def build_s3_key(s3_prefix: str, relative_key: str) -> str:
    """Build an S3 key from prefix and relative key."""
    return f"{normalize_prefix(s3_prefix=s3_prefix)}/{relative_key.strip('/')}"


def require_env_string(name: str) -> str:
    """Read a required environment variable as a non-empty string."""
    value = os.getenv(name)
    if not isinstance(value, str) or not value.strip():
        raise CloudTrainEvalError(f"Missing required environment variable: {name}")

    return value.strip()


def parse_float_env(name: str, default: float) -> float:
    """Read an optional environment variable as a float."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    try:
        return float(value)
    except ValueError as error:
        raise CloudTrainEvalError(f"{name} must be a float.") from error


def parse_int_env(name: str, default: int) -> int:
    """Read an optional environment variable as an integer."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    try:
        return int(value)
    except ValueError as error:
        raise CloudTrainEvalError(f"{name} must be an integer.") from error


def load_config_from_env() -> CloudTrainEvalConfig:
    """Load cloud training/evaluation configuration from environment variables."""
    dataset_mode = os.getenv(
        "COLLIDEROPS_DATASET_MODE",
        DEFAULT_DATASET_MODE,
    ).strip().lower()

    if dataset_mode not in SUPPORTED_DATASET_MODES:
        raise CloudTrainEvalError(
            f"Unsupported dataset_mode={dataset_mode}. "
            f"Supported={sorted(SUPPORTED_DATASET_MODES)}"
        )

    s3_bucket = require_env_string("COLLIDEROPS_S3_BUCKET")
    s3_prefix = normalize_prefix(os.getenv("COLLIDEROPS_S3_PREFIX", DEFAULT_S3_PREFIX))
    run_id = os.getenv(
        "COLLIDEROPS_RUN_ID",
        f"{dataset_mode}-{utc_timestamp()}",
    ).strip()
    test_size = parse_float_env("COLLIDEROPS_TEST_SIZE", DEFAULT_TEST_SIZE)
    random_state = parse_int_env("COLLIDEROPS_RANDOM_STATE", DEFAULT_RANDOM_STATE)

    if not 0 < test_size < 1:
        raise CloudTrainEvalError("COLLIDEROPS_TEST_SIZE must be between 0 and 1.")

    return CloudTrainEvalConfig(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        dataset_mode=dataset_mode,
        run_id=run_id,
        test_size=test_size,
        random_state=random_state,
    )


def build_curated_dataset_uri(config: CloudTrainEvalConfig) -> str:
    """Build the curated Parquet dataset S3 URI."""
    key = build_s3_key(
        s3_prefix=config.s3_prefix,
        relative_key=f"curated/{config.dataset_mode}/training_dataset.parquet/",
    )
    return build_s3_uri(bucket=config.s3_bucket, key=key)


def build_model_key(config: CloudTrainEvalConfig, model_name: str) -> str:
    """Build a model artifact S3 key."""
    return build_s3_key(
        s3_prefix=config.s3_prefix,
        relative_key=f"models/{config.dataset_mode}/{config.run_id}/{model_name}.joblib",
    )


def build_evaluation_key(config: CloudTrainEvalConfig, filename: str) -> str:
    """Build an evaluation artifact S3 key."""
    return build_s3_key(
        s3_prefix=config.s3_prefix,
        relative_key=f"evaluation/{config.dataset_mode}/{config.run_id}/{filename}",
    )


def load_training_data(config: CloudTrainEvalConfig) -> pd.DataFrame:
    """Read curated training data from S3 Parquet."""
    curated_dataset_uri = build_curated_dataset_uri(config=config)
    print(f"Loading curated dataset from {curated_dataset_uri}")

    try:
        return pd.read_parquet(curated_dataset_uri)
    except Exception as error:
        raise CloudTrainEvalError(
            f"Failed to read curated dataset from {curated_dataset_uri}: {error}"
        ) from error


def validate_training_data(df: pd.DataFrame) -> list[str]:
    """Validate the curated dataset and return feature columns."""
    if df.empty:
        raise CloudTrainEvalError("Curated training dataset is empty.")

    if LABEL_COLUMN not in df.columns:
        raise CloudTrainEvalError(
            f"Curated training dataset is missing label column: {LABEL_COLUMN}"
        )

    label_counts = df[LABEL_COLUMN].value_counts().to_dict()
    missing_labels = {NEGATIVE_LABEL, POSITIVE_LABEL} - set(label_counts)
    if missing_labels:
        raise CloudTrainEvalError(
            f"Curated training dataset is missing labels: {sorted(missing_labels)}"
        )

    feature_columns = [column for column in df.columns if column != LABEL_COLUMN]
    if not feature_columns:
        raise CloudTrainEvalError("Curated training dataset has no feature columns.")

    non_numeric_columns = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric_columns:
        raise CloudTrainEvalError(
            f"Feature columns must be numeric: {non_numeric_columns}"
        )

    null_feature_count = int(df[feature_columns].isnull().sum().sum())
    if null_feature_count != 0:
        raise CloudTrainEvalError(
            f"Feature columns contain null values: {null_feature_count}"
        )

    print(
        json.dumps(
            {
                "row_count": int(len(df)),
                "feature_count": len(feature_columns),
                "label_counts": {
                    str(key): int(value) for key, value in label_counts.items()
                },
            },
            indent=2,
        )
    )

    return feature_columns


def build_models(random_state: int) -> dict[str, Any]:
    """Build baseline sklearn models for cloud training."""
    return {
        "logistic_regression_curated_higgs": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1_000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest_curated_higgs": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting_curated_higgs": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=150,
            random_state=random_state,
        ),
    }


def get_positive_class_scores(model: Any, x_test: pd.DataFrame) -> list[float] | None:
    """Return probability scores for ROC-AUC when the model supports them."""
    if not hasattr(model, "predict_proba"):
        return None

    classes = list(model.classes_)
    if POSITIVE_LABEL not in classes:
        return None

    positive_class_index = classes.index(POSITIVE_LABEL)
    probabilities = model.predict_proba(x_test)
    return probabilities[:, positive_class_index].tolist()


def evaluate_model(
    model_name: str,
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Evaluate a trained sklearn model."""
    predictions = model.predict(x_test)
    positive_scores = get_positive_class_scores(model=model, x_test=x_test)

    metrics = {
        "model_name": model_name,
        "recall_score": recall_score(y_test, predictions, pos_label=POSITIVE_LABEL),
        "precision_score": precision_score(
            y_test,
            predictions,
            pos_label=POSITIVE_LABEL,
            zero_division=0,
        ),
        "accuracy_score": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions, pos_label=POSITIVE_LABEL),
        "roc_auc_score": None,
    }

    if positive_scores is not None:
        metrics["roc_auc_score"] = roc_auc_score(y_test, positive_scores)

    return metrics


def upload_file_to_s3(
    s3_client: Any,
    bucket: str,
    local_path: Path,
    key: str,
    content_type: str | None = None,
) -> None:
    """Upload a local file to S3."""
    extra_args = {"ContentType": content_type} if content_type else None
    try:
        if extra_args:
            s3_client.upload_file(str(local_path), bucket, key, ExtraArgs=extra_args)
        else:
            s3_client.upload_file(str(local_path), bucket, key)
    except ClientError as error:
        raise CloudTrainEvalError(
            f"Failed to upload {local_path} to s3://{bucket}/{key}: {error}"
        ) from error


def save_model_to_s3(
    model: Any,
    model_name: str,
    config: CloudTrainEvalConfig,
    s3_client: Any,
    output_dir: Path,
) -> str:
    """Save a trained model locally and upload it to S3."""
    local_model_path = output_dir / f"{model_name}.joblib"
    joblib.dump(model, local_model_path)

    model_key = build_model_key(config=config, model_name=model_name)
    upload_file_to_s3(
        s3_client=s3_client,
        bucket=config.s3_bucket,
        local_path=local_model_path,
        key=model_key,
        content_type="application/octet-stream",
    )
    return model_key


def build_evaluation_summary(
    config: CloudTrainEvalConfig,
    metrics_df: pd.DataFrame,
    model_keys: dict[str, str],
) -> str:
    """Build a Markdown evaluation summary."""
    best_by_roc_auc = metrics_df.sort_values("roc_auc_score", ascending=False).iloc[0]
    best_by_f1 = metrics_df.sort_values("f1_score", ascending=False).iloc[0]

    lines = [
        f"# ColliderOpsAI Evaluation Summary - {config.dataset_mode}",
        "",
        f"Run ID: `{config.run_id}`",
        f"Created at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Best models",
        "",
        f"- Best ROC-AUC: `{best_by_roc_auc['model_name']}` = "
        f"`{best_by_roc_auc['roc_auc_score']:.6f}`",
        f"- Best F1: `{best_by_f1['model_name']}` = "
        f"`{best_by_f1['f1_score']:.6f}`",
        "",
        "## Model artifact keys",
        "",
    ]

    for model_name, model_key in model_keys.items():
        lines.append(f"- `{model_name}`: `s3://{config.s3_bucket}/{model_key}`")

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            metrics_df.to_markdown(index=False),
            "",
        ]
    )

    return "\n".join(lines)


def run_cloud_train_eval() -> dict[str, Any]:
    """Run cloud training/evaluation and publish artifacts to S3."""
    config = load_config_from_env()
    s3_client = boto3.client("s3")

    df = load_training_data(config=config)
    feature_columns = validate_training_data(df=df)

    x = df[feature_columns]
    y = df[LABEL_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    models = build_models(random_state=config.random_state)
    metrics_rows: list[dict[str, Any]] = []
    model_keys: dict[str, str] = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        for model_name, model in models.items():
            print(f"Training model: {model_name}")
            model.fit(x_train, y_train)

            print(f"Evaluating model: {model_name}")
            metrics = evaluate_model(
                model_name=model_name,
                model=model,
                x_test=x_test,
                y_test=y_test,
            )
            metrics_rows.append(metrics)

            model_key = save_model_to_s3(
                model=model,
                model_name=model_name,
                config=config,
                s3_client=s3_client,
                output_dir=output_dir,
            )
            model_keys[model_name] = model_key

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_path = output_dir / "model_comparison.csv"
        metrics_df.to_csv(metrics_path, index=False)

        summary_text = build_evaluation_summary(
            config=config,
            metrics_df=metrics_df,
            model_keys=model_keys,
        )
        summary_path = output_dir / "evaluation_summary.md"
        summary_path.write_text(summary_text, encoding="utf-8")

        model_comparison_key = build_evaluation_key(
            config=config,
            filename="model_comparison.csv",
        )
        evaluation_summary_key = build_evaluation_key(
            config=config,
            filename="evaluation_summary.md",
        )

        upload_file_to_s3(
            s3_client=s3_client,
            bucket=config.s3_bucket,
            local_path=metrics_path,
            key=model_comparison_key,
            content_type="text/csv",
        )
        upload_file_to_s3(
            s3_client=s3_client,
            bucket=config.s3_bucket,
            local_path=summary_path,
            key=evaluation_summary_key,
            content_type="text/markdown",
        )

    result = {
        "status": "success",
        "dataset_mode": config.dataset_mode,
        "run_id": config.run_id,
        "s3_bucket": config.s3_bucket,
        "s3_prefix": config.s3_prefix,
        "feature_count": len(feature_columns),
        "model_count": len(models),
        "model_keys": model_keys,
        "model_comparison_key": model_comparison_key,
        "evaluation_summary_key": evaluation_summary_key,
        "model_comparison_uri": build_s3_uri(config.s3_bucket, model_comparison_key),
        "evaluation_summary_uri": build_s3_uri(
            config.s3_bucket,
            evaluation_summary_key,
        ),
    }

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_cloud_train_eval()