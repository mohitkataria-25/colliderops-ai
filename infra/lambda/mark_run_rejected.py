

"""Mark a ColliderOpsAI pipeline run as rejected.

This Lambda runs when evaluation gates fail. It writes a rejection record to S3
so the pipeline has an auditable failure artifact instead of silently dropping a
failed model run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError


DEFAULT_S3_PREFIX = "dev"


class MarkRunRejectedError(ValueError):
    """Raised when a rejected run cannot be recorded."""


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
        raise MarkRunRejectedError(f"Missing or invalid required field: {field_name}")

    return value.strip()


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
        raise MarkRunRejectedError(
            f"Failed to write rejection record to s3://{bucket}/{key}: {error}"
        ) from error


def build_rejection_key(dataset_mode: str, run_id: str, s3_prefix: str) -> str:
    """Build the S3 key for a rejected-run record."""
    return build_s3_key(
        s3_prefix=s3_prefix,
        relative_key=f"model_registry/{dataset_mode}/{run_id}/rejected_run.json",
    )


def mark_run_rejected(
    event: dict[str, Any],
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Write a rejected-run record to S3."""
    dataset_mode = require_string_field(event, "dataset_mode")
    run_id = require_string_field(event, "run_id")
    s3_bucket = require_string_field(event, "s3_bucket")
    s3_prefix = normalize_prefix(event.get("s3_prefix"))
    evaluation_gate_result = event.get("evaluation_gate_result", {})

    rejected_at = datetime.now(timezone.utc).isoformat()
    rejection_key = build_rejection_key(
        dataset_mode=dataset_mode,
        run_id=run_id,
        s3_prefix=s3_prefix,
    )

    rejection_record = {
        "dataset_mode": dataset_mode,
        "run_id": run_id,
        "status": "rejected",
        "reason": "evaluation_gates_failed",
        "evaluation_gate_result": evaluation_gate_result,
        "rejected_at": rejected_at,
    }

    s3_client = s3_client or boto3.client("s3")
    put_json_to_s3(
        s3_client=s3_client,
        bucket=s3_bucket,
        key=rejection_key,
        payload=rejection_record,
    )

    return {
        "marked_rejected": True,
        "dataset_mode": dataset_mode,
        "run_id": run_id,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "rejection_key": rejection_key,
        "rejection_uri": f"s3://{s3_bucket}/{rejection_key}",
        "rejected_at": rejected_at,
    }


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda entrypoint."""
    try:
        payload = mark_run_rejected(event=event)
        return build_response(status_code=200, body=payload)
    except MarkRunRejectedError as error:
        return build_response(
            status_code=400,
            body={
                "marked_rejected": False,
                "error_type": "MarkRunRejectedError",
                "message": str(error),
            },
        )
    except Exception as error:
        return build_response(
            status_code=500,
            body={
                "marked_rejected": False,
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
            "passed": False,
            "best_f1_score": 0.61,
            "best_roc_auc_score": 0.70,
        },
    }
    print(json.dumps(lambda_handler(sample_event, None), indent=2))