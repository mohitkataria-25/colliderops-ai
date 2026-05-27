

"""Publish the latest evaluation summary for a ColliderOpsAI dataset mode.

This Lambda copies a run-specific evaluation summary to stable "latest" paths.
The latest paths can be used by dashboards, RAG ingestion, documentation refresh,
or simple portfolio demos.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError


DEFAULT_S3_PREFIX = "dev"


class PublishEvaluationSummaryError(ValueError):
    """Raised when evaluation summary publishing fails."""


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
        raise PublishEvaluationSummaryError(
            f"Missing or invalid required field: {field_name}"
        )

    return value.strip()


def copy_s3_object(
    s3_client: Any,
    bucket: str,
    source_key: str,
    destination_key: str,
) -> None:
    """Copy an S3 object within the same bucket."""
    try:
        s3_client.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": source_key},
            Key=destination_key,
            ContentType="text/markdown",
            MetadataDirective="REPLACE",
        )
    except ClientError as error:
        raise PublishEvaluationSummaryError(
            "Failed to publish evaluation summary from "
            f"s3://{bucket}/{source_key} to s3://{bucket}/{destination_key}: {error}"
        ) from error


def publish_evaluation_summary(
    event: dict[str, Any],
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Copy run evaluation summary to latest evaluation and RAG source paths."""
    dataset_mode = require_string_field(event, "dataset_mode")
    run_id = require_string_field(event, "run_id")
    s3_bucket = require_string_field(event, "s3_bucket")
    s3_prefix = normalize_prefix(event.get("s3_prefix"))

    source_key = build_s3_key(
        s3_prefix=s3_prefix,
        relative_key=f"evaluation/{dataset_mode}/{run_id}/evaluation_summary.md",
    )
    latest_key = build_s3_key(
        s3_prefix=s3_prefix,
        relative_key=f"evaluation/latest/{dataset_mode}/evaluation_summary.md",
    )
    rag_source_key = build_s3_key(
        s3_prefix=s3_prefix,
        relative_key=f"rag/source_docs/evaluation_summary_{dataset_mode}.md",
    )

    s3_client = s3_client or boto3.client("s3")
    copy_s3_object(
        s3_client=s3_client,
        bucket=s3_bucket,
        source_key=source_key,
        destination_key=latest_key,
    )
    copy_s3_object(
        s3_client=s3_client,
        bucket=s3_bucket,
        source_key=source_key,
        destination_key=rag_source_key,
    )

    published_at = datetime.now(timezone.utc).isoformat()
    return {
        "published": True,
        "dataset_mode": dataset_mode,
        "run_id": run_id,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "source_key": source_key,
        "latest_key": latest_key,
        "rag_source_key": rag_source_key,
        "source_uri": f"s3://{s3_bucket}/{source_key}",
        "latest_uri": f"s3://{s3_bucket}/{latest_key}",
        "rag_source_uri": f"s3://{s3_bucket}/{rag_source_key}",
        "published_at": published_at,
    }


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda entrypoint."""
    try:
        payload = publish_evaluation_summary(event=event)
        return build_response(status_code=200, body=payload)
    except PublishEvaluationSummaryError as error:
        return build_response(
            status_code=400,
            body={
                "published": False,
                "error_type": "PublishEvaluationSummaryError",
                "message": str(error),
            },
        )
    except Exception as error:
        return build_response(
            status_code=500,
            body={
                "published": False,
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