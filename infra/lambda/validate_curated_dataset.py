"""Validate curated dataset outputs for the ColliderOpsAI AWS pipeline.

This Lambda runs after ETL and before model training. It validates that the
curated dataset artifact exists in S3 and that its metadata indicates the dataset
is ready for model training.

This Lambda intentionally reads metadata JSON only. It does not read the full
curated dataset because Lambda should stay lightweight.
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

EXPECTED_FEATURE_COUNTS = {
    "sample_collider": 5,
    "real_cern": 2,
    "curated_higgs": 28,
}

MINIMUM_ROW_COUNTS = {
    "sample_collider": 2,
    "real_cern": 100,
    "curated_higgs": 1_000,
}

REQUIRED_LABELS = {"background", "signal"}


class CuratedDatasetValidationError(ValueError):
    """Raised when curated dataset validation fails."""


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
        raise CuratedDatasetValidationError(
            f"Missing or invalid required field: {field_name}"
        )

    return value.strip()


def validate_dataset_mode(dataset_mode: str) -> str:
    """Validate and normalize dataset mode."""
    normalized_dataset_mode = dataset_mode.strip().lower()

    if normalized_dataset_mode not in SUPPORTED_DATASET_MODES:
        raise CuratedDatasetValidationError(
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
    """Return True if an S3 object or prefix exists, otherwise False.

    Spark writes Parquet outputs as a directory-like S3 prefix, for example:
    dev/curated/curated_higgs/training_dataset.parquet/

    `head_object` only works for real objects, not prefixes. So this function
    first checks for a direct object and then falls back to checking whether any
    objects exist under the prefix.
    """
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as error:
        error_code = str(error.response.get("Error", {}).get("Code"))
        if error_code not in {"404", "NoSuchKey", "NotFound"}:
            raise

    prefix = key if key.endswith("/") else f"{key}/"
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        MaxKeys=1,
    )
    return response.get("KeyCount", 0) > 0


def read_json_from_s3(
    s3_client: Any,
    bucket: str,
    key: str,
) -> dict[str, Any]:
    """Read a JSON object from S3."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
    except ClientError as error:
        error_code = str(error.response.get("Error", {}).get("Code"))
        if error_code in {"404", "NoSuchKey", "NoSuchBucket", "NotFound"}:
            raise CuratedDatasetValidationError(
                f"Required metadata object not found: s3://{bucket}/{key}"
            ) from error
        raise

    body = response["Body"].read().decode("utf-8")
    return json.loads(body)


def build_curated_dataset_key(dataset_mode: str, s3_prefix: str) -> str:
    """Build expected curated dataset key for a dataset mode."""
    return build_s3_key(
        s3_prefix=s3_prefix,
        relative_key=f"curated/{dataset_mode}/training_dataset.parquet",
    )


def build_curated_metadata_key(dataset_mode: str, s3_prefix: str) -> str:
    """Build expected curated metadata key for a dataset mode."""
    return build_s3_key(
        s3_prefix=s3_prefix,
        relative_key=f"curated/{dataset_mode}/dataset_metadata.json",
    )


def validate_label_counts(label_counts: dict[str, Any]) -> None:
    """Validate label-count metadata."""
    if not isinstance(label_counts, dict) or not label_counts:
        raise CuratedDatasetValidationError("label_counts must be a non-empty object.")

    available_labels = set(label_counts)
    missing_labels = sorted(REQUIRED_LABELS - available_labels)
    if missing_labels:
        raise CuratedDatasetValidationError(
            f"Curated dataset is missing required labels: {missing_labels}"
        )

    for label, count in label_counts.items():
        try:
            normalized_count = int(count)
        except (TypeError, ValueError) as error:
            raise CuratedDatasetValidationError(
                f"Label count for {label} must be an integer. Received={count}"
            ) from error

        if normalized_count <= 0:
            raise CuratedDatasetValidationError(
                f"Label count for {label} must be positive. Received={count}"
            )


def validate_metadata_values(
    dataset_mode: str,
    metadata: dict[str, Any],
) -> None:
    """Validate curated dataset metadata values."""
    expected_feature_count = EXPECTED_FEATURE_COUNTS[dataset_mode]
    minimum_row_count = MINIMUM_ROW_COUNTS[dataset_mode]

    row_count = int(metadata.get("row_count", 0))
    feature_count = int(metadata.get("feature_count", 0))
    null_feature_count = int(metadata.get("null_feature_count", 0))

    if row_count < minimum_row_count:
        raise CuratedDatasetValidationError(
            f"row_count too low for {dataset_mode}. "
            f"Expected at least {minimum_row_count}, received {row_count}."
        )

    if feature_count != expected_feature_count:
        raise CuratedDatasetValidationError(
            f"feature_count mismatch for {dataset_mode}. "
            f"Expected {expected_feature_count}, received {feature_count}."
        )

    if not bool(metadata.get("schema_valid", False)):
        raise CuratedDatasetValidationError("schema_valid must be true.")

    if not bool(metadata.get("features_numeric", False)):
        raise CuratedDatasetValidationError("features_numeric must be true.")

    if not bool(metadata.get("two_class_training_ready", False)):
        raise CuratedDatasetValidationError("two_class_training_ready must be true.")

    if null_feature_count != 0:
        raise CuratedDatasetValidationError(
            f"null_feature_count must be 0. Received={null_feature_count}."
        )

    validate_label_counts(metadata.get("label_counts", {}))


def validate_curated_dataset(
    event: dict[str, Any],
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Validate curated dataset output and metadata in S3."""
    dataset_mode = validate_dataset_mode(
        dataset_mode=require_string_field(event, "dataset_mode")
    )
    run_id = require_string_field(event, "run_id")
    s3_bucket = require_string_field(event, "s3_bucket")
    s3_prefix = normalize_prefix(event.get("s3_prefix"))

    s3_client = s3_client or boto3.client("s3")

    curated_dataset_key = build_curated_dataset_key(
        dataset_mode=dataset_mode,
        s3_prefix=s3_prefix,
    )
    curated_metadata_key = build_curated_metadata_key(
        dataset_mode=dataset_mode,
        s3_prefix=s3_prefix,
    )

    curated_dataset_exists = check_s3_object_exists(
        s3_client=s3_client,
        bucket=s3_bucket,
        key=curated_dataset_key,
    )
    if not curated_dataset_exists:
        raise CuratedDatasetValidationError(
            f"Curated dataset not found: s3://{s3_bucket}/{curated_dataset_key}"
        )

    metadata = read_json_from_s3(
        s3_client=s3_client,
        bucket=s3_bucket,
        key=curated_metadata_key,
    )
    validate_metadata_values(dataset_mode=dataset_mode, metadata=metadata)

    return {
        "validated": True,
        "dataset_mode": dataset_mode,
        "run_id": run_id,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "curated_dataset_key": curated_dataset_key,
        "curated_metadata_key": curated_metadata_key,
        "curated_dataset_uri": f"s3://{s3_bucket}/{curated_dataset_key}",
        "curated_metadata_uri": f"s3://{s3_bucket}/{curated_metadata_key}",
        "row_count": metadata.get("row_count"),
        "feature_count": metadata.get("feature_count"),
        "label_counts": metadata.get("label_counts"),
        "schema_valid": metadata.get("schema_valid"),
        "features_numeric": metadata.get("features_numeric"),
        "two_class_training_ready": metadata.get("two_class_training_ready"),
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }


def lambda_handler(
    event: dict[str, Any],
    context: Any,
) -> dict[str, Any]:
    """AWS Lambda entrypoint."""
    try:
        validated_payload = validate_curated_dataset(event=event)
        return build_response(
            status_code=200,
            body=validated_payload,
        )
    except CuratedDatasetValidationError as error:
        return build_response(
            status_code=400,
            body={
                "validated": False,
                "error_type": "CuratedDatasetValidationError",
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
    }
    print(json.dumps(lambda_handler(sample_event, None), indent=2))