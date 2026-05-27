"""Validate dataset metadata for the ColliderOpsAI AWS pipeline.

This Lambda is the second gate in the Step Functions pipeline. It receives the
normalized request from ValidatePipelineRequest and builds the expected S3 paths
for the selected dataset mode.

It performs lightweight metadata checks only. It does not run ETL, parse ROOT
files, train models, or read large datasets.
"""

from __future__ import annotations

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
CURATED_HIGGS_RAW_KEY = "raw/curated_higgs/HIGGS.csv.gz"
REAL_CERN_REGISTRY_KEY = "raw/real_cern/dataset_registry.json"
SAMPLE_COLLIDER_CURATED_KEY = "curated/sample_collider/training_dataset.parquet"


class DatasetMetadataValidationError(ValueError):
    """Raised when dataset metadata validation fails."""


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
        raise DatasetMetadataValidationError(
            f"Missing or invalid required field: {field_name}"
        )

    return value.strip()


def validate_dataset_mode(dataset_mode: str) -> str:
    """Validate and normalize dataset mode."""
    normalized_dataset_mode = dataset_mode.strip().lower()

    if normalized_dataset_mode not in SUPPORTED_DATASET_MODES:
        raise DatasetMetadataValidationError(
            "Unsupported dataset_mode. "
            f"Received={dataset_mode}. "
            f"Supported={sorted(SUPPORTED_DATASET_MODES)}"
        )

    return normalized_dataset_mode


def check_s3_object_exists(
    s3_client: Any,
    bucket: str,
    key: str,
) -> bool:
    """Return True if an S3 object exists, otherwise False."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as error:
        error_code = str(error.response.get("Error", {}).get("Code"))
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def build_expected_dataset_paths(
    dataset_mode: str,
    s3_prefix: str,
) -> dict[str, str | None]:
    """Build expected S3 keys for the requested dataset mode."""
    processed_key = f"processed/{dataset_mode}/run_metadata.json"
    curated_key = f"curated/{dataset_mode}/training_dataset.parquet"

    source_key_by_mode = {
        "curated_higgs": CURATED_HIGGS_RAW_KEY,
        "real_cern": REAL_CERN_REGISTRY_KEY,
        "sample_collider": SAMPLE_COLLIDER_CURATED_KEY,
    }

    source_key = source_key_by_mode[dataset_mode]

    return {
        "source_key": build_s3_key(s3_prefix=s3_prefix, relative_key=source_key),
        "processed_metadata_key": build_s3_key(
            s3_prefix=s3_prefix,
            relative_key=processed_key,
        ),
        "curated_dataset_key": build_s3_key(
            s3_prefix=s3_prefix,
            relative_key=curated_key,
        ),
    }


def validate_dataset_metadata(
    event: dict[str, Any],
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Validate dataset metadata and expected source availability."""
    dataset_mode = validate_dataset_mode(
        dataset_mode=require_string_field(event, "dataset_mode")
    )
    s3_bucket = require_string_field(event, "s3_bucket")
    s3_prefix = normalize_prefix(event.get("s3_prefix"))
    run_id = require_string_field(event, "run_id")

    s3_client = s3_client or boto3.client("s3")
    expected_paths = build_expected_dataset_paths(
        dataset_mode=dataset_mode,
        s3_prefix=s3_prefix,
    )

    source_exists = check_s3_object_exists(
        s3_client=s3_client,
        bucket=s3_bucket,
        key=str(expected_paths["source_key"]),
    )

    # For curated_higgs, the pipeline can download the public source if the raw
    # object has not been staged yet. For real_cern, the registry is required.
    if dataset_mode == "real_cern" and not source_exists:
        raise DatasetMetadataValidationError(
            "Missing required real_cern dataset registry in S3. "
            f"Expected s3://{s3_bucket}/{expected_paths['source_key']}"
        )

    if dataset_mode == "sample_collider" and not source_exists:
        raise DatasetMetadataValidationError(
            "Missing required sample_collider curated dataset in S3. "
            f"Expected s3://{s3_bucket}/{expected_paths['source_key']}"
        )

    return {
        "validated": True,
        "dataset_mode": dataset_mode,
        "run_id": run_id,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "source_exists": source_exists,
        "source_key": expected_paths["source_key"],
        "processed_metadata_key": expected_paths["processed_metadata_key"],
        "curated_dataset_key": expected_paths["curated_dataset_key"],
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }


def lambda_handler(
    event: dict[str, Any],
    context: Any,
) -> dict[str, Any]:
    """AWS Lambda entrypoint."""
    try:
        validated_payload = validate_dataset_metadata(event=event)
        return build_response(
            status_code=200,
            body=validated_payload,
        )
    except DatasetMetadataValidationError as error:
        return build_response(
            status_code=400,
            body={
                "validated": False,
                "error_type": "DatasetMetadataValidationError",
                "message": str(error),
            },
        )
    except Exception as error:
        return build_response(
            status_code=500,
            body={
                "validated": False,
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
        "total_rows": 10_000,
    }
    print(json.dumps(lambda_handler(sample_event, None), indent=2))