

"""Validate ColliderOpsAI pipeline request payload.

This Lambda is the first gate in the AWS Step Functions pipeline. It validates
basic execution inputs before the workflow starts dataset metadata checks, ETL,
training, and evaluation.

Expected event example:

{
    "dataset_mode": "curated_higgs",
    "run_type": "manual",
    "total_rows": 10000,
    "s3_bucket": "colliderops-ai-dev",
    "s3_prefix": "dev",
    "train_models": true,
    "evaluate_models": true
}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


SUPPORTED_DATASET_MODES = {
    "sample_collider",
    "real_cern",
    "curated_higgs",
}

SUPPORTED_RUN_TYPES = {
    "manual",
    "scheduled",
    "s3_event",
}

DEFAULT_S3_PREFIX = "dev"
DEFAULT_RUN_TYPE = "manual"
DEFAULT_TOTAL_ROWS = 10_000


class PipelineRequestValidationError(ValueError):
    """Raised when the pipeline request payload is invalid."""


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


def generate_run_id(dataset_mode: str) -> str:
    """Generate a deterministic-looking run id using UTC timestamp."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{dataset_mode}-{timestamp}"


def require_string_field(
    event: dict[str, Any],
    field_name: str,
) -> str:
    """Read and validate a required string field."""
    value = event.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise PipelineRequestValidationError(
            f"Missing or invalid required field: {field_name}"
        )

    return value.strip()


def validate_dataset_mode(dataset_mode: str) -> str:
    """Validate dataset mode."""
    normalized_dataset_mode = dataset_mode.strip().lower()

    if normalized_dataset_mode not in SUPPORTED_DATASET_MODES:
        raise PipelineRequestValidationError(
            "Unsupported dataset_mode. "
            f"Received={dataset_mode}. "
            f"Supported={sorted(SUPPORTED_DATASET_MODES)}"
        )

    return normalized_dataset_mode


def validate_run_type(run_type: str | None) -> str:
    """Validate run type."""
    normalized_run_type = (run_type or DEFAULT_RUN_TYPE).strip().lower()

    if normalized_run_type not in SUPPORTED_RUN_TYPES:
        raise PipelineRequestValidationError(
            "Unsupported run_type. "
            f"Received={run_type}. "
            f"Supported={sorted(SUPPORTED_RUN_TYPES)}"
        )

    return normalized_run_type


def validate_total_rows(
    dataset_mode: str,
    total_rows: Any,
) -> int | None:
    """Validate total_rows for dataset modes that use sampled tabular data."""
    if dataset_mode != "curated_higgs":
        return None

    if total_rows is None:
        total_rows = DEFAULT_TOTAL_ROWS

    try:
        normalized_total_rows = int(total_rows)
    except (TypeError, ValueError) as error:
        raise PipelineRequestValidationError(
            "total_rows must be an integer for curated_higgs runs."
        ) from error

    if normalized_total_rows < 2:
        raise PipelineRequestValidationError("total_rows must be at least 2.")

    if normalized_total_rows % 2 != 0:
        raise PipelineRequestValidationError(
            "total_rows must be an even number for balanced curated_higgs sampling."
        )

    return normalized_total_rows


def validate_boolean_field(
    event: dict[str, Any],
    field_name: str,
    default: bool,
) -> bool:
    """Validate optional boolean field."""
    value = event.get(field_name, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized_value = value.strip().lower()
        if normalized_value in {"true", "1", "yes"}:
            return True
        if normalized_value in {"false", "0", "no"}:
            return False

    raise PipelineRequestValidationError(
        f"{field_name} must be boolean or a boolean-like string."
    )


def validate_pipeline_request(event: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a Step Functions pipeline request."""
    dataset_mode = validate_dataset_mode(
        dataset_mode=require_string_field(event, "dataset_mode")
    )
    run_type = validate_run_type(event.get("run_type"))
    s3_bucket = require_string_field(event, "s3_bucket")
    s3_prefix = str(event.get("s3_prefix", DEFAULT_S3_PREFIX)).strip() or DEFAULT_S3_PREFIX
    total_rows = validate_total_rows(
        dataset_mode=dataset_mode,
        total_rows=event.get("total_rows"),
    )
    train_models = validate_boolean_field(
        event=event,
        field_name="train_models",
        default=True,
    )
    evaluate_models = validate_boolean_field(
        event=event,
        field_name="evaluate_models",
        default=True,
    )

    run_id = event.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        run_id = generate_run_id(dataset_mode=dataset_mode)
    else:
        run_id = run_id.strip()

    return {
        "validated": True,
        "dataset_mode": dataset_mode,
        "run_type": run_type,
        "run_id": run_id,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "total_rows": total_rows,
        "train_models": train_models,
        "evaluate_models": evaluate_models,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }


def lambda_handler(
        event:dict[str, Any],
        context:Any,
)-> dict[str, Any]:
    """AWS Lambda entrypoint."""

    try:
        validated_payload = validate_pipeline_request(event=event)
        return build_response(
            status_code=200,
            body=validated_payload,
        )
    except PipelineRequestValidationError as error:
        return build_response(
            status_code=400,
            body={
                "validated": False,
                "error_type":"PipelineRequestValidationError",
                "message": str(error),
            },
        )
    except Exception as error:
        return build_response(
            status_code=500,
            body={
                "validated": False,
                "error_type":type(error).__name__,
                "message":str(error),
            },
        )


if __name__ == "__main__":
    sample_event = {
        "dataset_mode": "curated_higgs",
        "run_type": "manual",
        "total_rows": 10_000,
        "s3_bucket": "colliderops-ai-dev",
        "s3_prefix": "dev",
        "train_models": True,
        "evaluate_models": True,
    }
    print(json.dumps(lambda_handler(sample_event, None), indent=2))