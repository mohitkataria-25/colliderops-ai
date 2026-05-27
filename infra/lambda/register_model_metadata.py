

"""Register accepted model metadata for the ColliderOpsAI AWS pipeline.

This Lambda runs after evaluation gates pass. It writes a lightweight model
registry record to S3 so downstream services can discover the best candidate
model and its evaluation artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError


DEFAULT_S3_PREFIX = "dev"
DEFAULT_REGISTRY_STATUS = "candidate"


class ModelMetadataRegistrationError(ValueError):
    """Raised when model metadata registration fails."""


def build_response(status_code: int, body: dict[str, Any]) -> dict[str, Any]:
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


def build_s3_key(s3_prefix: str, relative_key: str) -> str:
    """Build an S3 key using the environment prefix and a relative key."""
    return f"{normalize_prefix(s3_prefix)}/{relative_key.strip('/')}"


def require_string_field(event: dict[str, Any], field_name: str) -> str:
    """Read and validate a required string field."""
    value = event.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ModelMetadataRegistrationError(
            f"Missing or invalid required field: {field_name}"
        )

    return value.strip()


def require_dict_field(event: dict[str, Any], field_name: str) -> dict[str, Any]:
    """Read and validate a required dictionary field."""
    value = event.get(field_name)
    if not isinstance(value, dict) or not value:
        raise ModelMetadataRegistrationError(
            f"Missing or invalid required object field: {field_name}"
        )

    return value


def build_registry_keys(dataset_mode: str, run_id: str, s3_prefix: str) -> dict[str, str]:
    """Build registry keys for run-specific and latest model metadata."""
    run_registry_key = build_s3_key(
        s3_prefix=s3_prefix,
        relative_key=f"model_registry/{dataset_mode}/{run_id}/model_metadata.json",
    )
    latest_registry_key = build_s3_key(
        s3_prefix=s3_prefix,
        relative_key=f"model_registry/{dataset_mode}/latest/model_metadata.json",
    )

    return {
        "run_registry_key": run_registry_key,
        "latest_registry_key": latest_registry_key,
    }


def build_model_artifact_key(
    dataset_mode: str,
    model_name: str,
    run_id: str,
    s3_prefix: str,
) -> str:
    """Build expected model artifact key for the selected model."""
    return build_s3_key(
        s3_prefix=s3_prefix,
        relative_key=f"models/{dataset_mode}/{run_id}/{model_name}.joblib",
    )


def put_json_to_s3(
    s3_client: Any,
    bucket: str,
    key: str,
    payload: dict[str, Any],
) -> None:
    """Write a JSON payload to S3."""
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(payload, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
    except ClientError as error:
        raise ModelMetadataRegistrationError(
            f"Failed to write model metadata to s3://{bucket}/{key}: {error}"
        ) from error


def register_model_metadata(
    event: dict[str, Any],
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Register accepted model metadata in S3."""
    dataset_mode = require_string_field(event, "dataset_mode")
    run_id = require_string_field(event, "run_id")
    s3_bucket = require_string_field(event, "s3_bucket")
    s3_prefix = normalize_prefix(event.get("s3_prefix"))
    evaluation_gate_result = require_dict_field(event, "evaluation_gate_result")

    if not bool(evaluation_gate_result.get("passed", False)):
        raise ModelMetadataRegistrationError(
            "Cannot register model metadata because evaluation gates did not pass."
        )

    best_model_name = str(evaluation_gate_result.get("best_model_by_roc_auc") or "").strip()
    if not best_model_name:
        raise ModelMetadataRegistrationError(
            "evaluation_gate_result is missing best_model_by_roc_auc."
        )

    registry_keys = build_registry_keys(
        dataset_mode=dataset_mode,
        run_id=run_id,
        s3_prefix=s3_prefix,
    )
    model_artifact_key = build_model_artifact_key(
        dataset_mode=dataset_mode,
        model_name=best_model_name,
        run_id=run_id,
        s3_prefix=s3_prefix,
    )
    model_comparison_key = build_s3_key(
        s3_prefix=s3_prefix,
        relative_key=f"evaluation/{dataset_mode}/{run_id}/model_comparison.csv",
    )
    evaluation_summary_key = build_s3_key(
        s3_prefix=s3_prefix,
        relative_key=f"evaluation/{dataset_mode}/{run_id}/evaluation_summary.md",
    )

    registered_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "dataset_mode": dataset_mode,
        "run_id": run_id,
        "status": DEFAULT_REGISTRY_STATUS,
        "selection_metric": "roc_auc_score",
        "model_name": best_model_name,
        "model_artifact_key": model_artifact_key,
        "model_artifact_uri": f"s3://{s3_bucket}/{model_artifact_key}",
        "model_comparison_key": model_comparison_key,
        "model_comparison_uri": f"s3://{s3_bucket}/{model_comparison_key}",
        "evaluation_summary_key": evaluation_summary_key,
        "evaluation_summary_uri": f"s3://{s3_bucket}/{evaluation_summary_key}",
        "best_f1_score": evaluation_gate_result.get("best_f1_score"),
        "best_roc_auc_score": evaluation_gate_result.get("best_roc_auc_score"),
        "best_accuracy_score": evaluation_gate_result.get("best_accuracy_score"),
        "evaluation_gate_result": evaluation_gate_result,
        "registered_at": registered_at,
    }

    s3_client = s3_client or boto3.client("s3")
    put_json_to_s3(
        s3_client=s3_client,
        bucket=s3_bucket,
        key=registry_keys["run_registry_key"],
        payload=metadata,
    )
    put_json_to_s3(
        s3_client=s3_client,
        bucket=s3_bucket,
        key=registry_keys["latest_registry_key"],
        payload=metadata,
    )

    return {
        "registered": True,
        "dataset_mode": dataset_mode,
        "run_id": run_id,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "model_name": best_model_name,
        "run_registry_key": registry_keys["run_registry_key"],
        "latest_registry_key": registry_keys["latest_registry_key"],
        "registered_at": registered_at,
    }


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda entrypoint."""
    try:
        payload = register_model_metadata(event=event)
        return build_response(status_code=200, body=payload)
    except ModelMetadataRegistrationError as error:
        return build_response(
            status_code=400,
            body={
                "registered": False,
                "error_type": "ModelMetadataRegistrationError",
                "message": str(error),
            },
        )
    except Exception as error:
        return build_response(
            status_code=500,
            body={
                "registered": False,
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
        "evaluation_gate_result": {
            "passed": True,
            "best_model_by_roc_auc": "hist_gradient_boosting_curated_higgs",
            "best_f1_score": 0.700,
            "best_roc_auc_score": 0.786,
            "best_accuracy_score": 0.704,
        },
    }
    print(json.dumps(lambda_handler(sample_event, None), indent=2))