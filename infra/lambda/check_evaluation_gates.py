

"""Check model evaluation gates for the ColliderOpsAI AWS pipeline.

This Lambda runs after the evaluation job. It reads the model comparison metrics
artifact from S3, identifies the best models, and decides whether the run should
be accepted as a candidate model run or rejected.

The Lambda is intentionally lightweight. It reads the model-comparison CSV only;
it does not load trained models or full datasets.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError


SUPPORTED_DATASET_MODES = {
    "sample_collider",
    "real_cern",
    "curated_higgs",
}

DEFAULT_S3_PREFIX = "dev"

# These are initial gates for portfolio/cloud-pipeline validation.
# Tune them as the datasets and models mature.
EVALUATION_GATES = {
    "sample_collider": {
        "minimum_f1_score": 0.50,
        "minimum_roc_auc_score": 0.50,
    },
    "real_cern": {
        "minimum_f1_score": 0.70,
        "minimum_roc_auc_score": 0.70,
    },
    "curated_higgs": {
        "minimum_f1_score": 0.68,
        "minimum_roc_auc_score": 0.75,
    },
}


class EvaluationGateError(ValueError):
    """Raised when evaluation gate checks fail."""


def build_response(
    status_code: int,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Build a Lambda-style response object."""
    return {
        "statusCode": status_code,
        "body": json.dumps(body),
        **body,
    }


def normalize_prefix(s3_prefix: str | None) -> str:
    """Normalize S3 prefix without leading/trailing slashes."""
    if not s3_prefix:
        return DEFAULT_S3_PREFIX

    return s3_prefix.strip().strip("/") or DEFAULT_S3_PREFIX


def build_s3_key(
    s3_prefix: str,
    relative_key: str,
) -> str:
    """Build an S3 key using the environment prefix and a relative key."""
    normalized_prefix = normalize_prefix(s3_prefix)
    return f"{normalized_prefix}/{relative_key.strip('/')}"


def require_string_field(
    event: dict[str, Any],
    field_name: str,
) -> str:
    """Read and validate a required string field."""
    value = event.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise EvaluationGateError(f"Missing or invalid required field: {field_name}")

    return value.strip()


def validate_dataset_mode(dataset_mode: str) -> str:
    """Validate and normalize dataset mode."""
    normalized_dataset_mode = dataset_mode.strip().lower()

    if normalized_dataset_mode not in SUPPORTED_DATASET_MODES:
        raise EvaluationGateError(
            "Unsupported dataset_mode. "
            f"Received={dataset_mode}. "
            f"Supported={sorted(SUPPORTED_DATASET_MODES)}"
        )

    return normalized_dataset_mode


def build_model_comparison_key(
    dataset_mode: str,
    run_id: str,
    s3_prefix: str,
) -> str:
    """Build expected model-comparison metrics key."""
    return build_s3_key(
        s3_prefix=s3_prefix,
        relative_key=f"evaluation/{dataset_mode}/{run_id}/model_comparison.csv",
    )


def build_evaluation_summary_key(
    dataset_mode: str,
    run_id: str,
    s3_prefix: str,
) -> str:
    """Build expected evaluation summary key."""
    return build_s3_key(
        s3_prefix=s3_prefix,
        relative_key=f"evaluation/{dataset_mode}/{run_id}/evaluation_summary.md",
    )


def read_text_from_s3(
    s3_client: Any,
    bucket: str,
    key: str,
) -> str:
    """Read a text object from S3."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
    except ClientError as error:
        error_code = str(error.response.get("Error", {}).get("Code"))
        if error_code in {"404", "NoSuchKey", "NoSuchBucket", "NotFound"}:
            raise EvaluationGateError(
                f"Required evaluation artifact not found: s3://{bucket}/{key}"
            ) from error
        raise

    return response["Body"].read().decode("utf-8")


def parse_model_comparison_csv(csv_text: str) -> list[dict[str, Any]]:
    """Parse model-comparison CSV text into metric rows."""
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)

    if not rows:
        raise EvaluationGateError("model_comparison.csv is empty.")

    required_columns = {
        "model_name",
        "recall_score",
        "precision_score",
        "accuracy_score",
        "f1_score",
        "roc_auc_score",
    }
    missing_columns = sorted(required_columns - set(rows[0]))
    if missing_columns:
        raise EvaluationGateError(
            f"model_comparison.csv is missing required columns: {missing_columns}"
        )

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        parsed_rows.append(
            {
                "model_name": row["model_name"],
                "recall_score": float(row["recall_score"]),
                "precision_score": float(row["precision_score"]),
                "accuracy_score": float(row["accuracy_score"]),
                "f1_score": float(row["f1_score"]),
                "roc_auc_score": float(row["roc_auc_score"]),
            }
        )

    return parsed_rows


def find_best_model(
    metric_rows: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    """Find the best model row by a metric."""
    return max(metric_rows, key=lambda row: row[metric_name])


def evaluate_gates(
    dataset_mode: str,
    metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply dataset-mode-specific evaluation gates."""
    gates = EVALUATION_GATES[dataset_mode]
    best_f1_row = find_best_model(metric_rows=metric_rows, metric_name="f1_score")
    best_roc_auc_row = find_best_model(
        metric_rows=metric_rows,
        metric_name="roc_auc_score",
    )
    best_accuracy_row = find_best_model(
        metric_rows=metric_rows,
        metric_name="accuracy_score",
    )

    best_f1 = best_f1_row["f1_score"]
    best_roc_auc = best_roc_auc_row["roc_auc_score"]

    f1_passed = best_f1 >= gates["minimum_f1_score"]
    roc_auc_passed = best_roc_auc >= gates["minimum_roc_auc_score"]
    passed = f1_passed and roc_auc_passed

    return {
        "passed": passed,
        "f1_passed": f1_passed,
        "roc_auc_passed": roc_auc_passed,
        "minimum_f1_score": gates["minimum_f1_score"],
        "minimum_roc_auc_score": gates["minimum_roc_auc_score"],
        "best_model_by_f1": best_f1_row["model_name"],
        "best_f1_score": best_f1,
        "best_model_by_roc_auc": best_roc_auc_row["model_name"],
        "best_roc_auc_score": best_roc_auc,
        "best_model_by_accuracy": best_accuracy_row["model_name"],
        "best_accuracy_score": best_accuracy_row["accuracy_score"],
    }


def check_evaluation_gates(
    event: dict[str, Any],
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Read evaluation metrics from S3 and apply quality gates."""
    dataset_mode = validate_dataset_mode(
        dataset_mode=require_string_field(event, "dataset_mode")
    )
    run_id = require_string_field(event, "run_id")
    s3_bucket = require_string_field(event, "s3_bucket")
    s3_prefix = normalize_prefix(event.get("s3_prefix"))

    s3_client = s3_client or boto3.client("s3")
    model_comparison_key = build_model_comparison_key(
        dataset_mode=dataset_mode,
        run_id=run_id,
        s3_prefix=s3_prefix,
    )
    evaluation_summary_key = build_evaluation_summary_key(
        dataset_mode=dataset_mode,
        run_id=run_id,
        s3_prefix=s3_prefix,
    )

    csv_text = read_text_from_s3(
        s3_client=s3_client,
        bucket=s3_bucket,
        key=model_comparison_key,
    )
    metric_rows = parse_model_comparison_csv(csv_text=csv_text)
    gate_result = evaluate_gates(dataset_mode=dataset_mode, metric_rows=metric_rows)

    return {
        "validated": True,
        "dataset_mode": dataset_mode,
        "run_id": run_id,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "model_comparison_key": model_comparison_key,
        "evaluation_summary_key": evaluation_summary_key,
        "model_count": len(metric_rows),
        "gate_result": gate_result,
        **gate_result,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def lambda_handler(
    event: dict[str, Any],
    context: Any,
) -> dict[str, Any]:
    """AWS Lambda entrypoint."""
    try:
        gate_payload = check_evaluation_gates(event=event)
        return build_response(status_code=200, body=gate_payload)
    except EvaluationGateError as error:
        return build_response(
            status_code=400,
            body={
                "validated": False,
                "passed": False,
                "error_type": "EvaluationGateError",
                "message": str(error),
            },
        )
    except Exception as error:
        return build_response(
            status_code=500,
            body={
                "validated": False,
                "passed": False,
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )


if __name__ == "__main__":
    sample_event = {
        "dataset_mode": "curated_higgs",
        "run_id": "curated_higgs-local-test",
        "s3_bucket": "colliderops-ai-dev",
        "s3_prefix": "dev",
    }
    print(json.dumps(lambda_handler(sample_event, None), indent=2))