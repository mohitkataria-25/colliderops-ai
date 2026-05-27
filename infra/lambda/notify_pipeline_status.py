

"""Notify ColliderOpsAI pipeline status.

This Lambda provides a lightweight notification hook for Step Functions success
and failure paths. It logs a structured notification payload and can optionally
publish to SNS when COLLIDEROPS_SNS_TOPIC_ARN is configured.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

import boto3
from botocore.exceptions import ClientError


DEFAULT_S3_PREFIX = "dev"
SNS_TOPIC_ARN_ENV_VAR = "COLLIDEROPS_SNS_TOPIC_ARN"
WRITE_NOTIFICATION_TO_S3_ENV_VAR = "COLLIDEROPS_WRITE_NOTIFICATION_TO_S3"
VALID_STATUSES = {"success", "failed", "rejected"}


class NotifyPipelineStatusError(ValueError):
    """Raised when pipeline status notification fails."""


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


def require_string_field(event: dict[str, Any], field_name: str) -> str:
    """Read and validate a required string field."""
    value = event.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise NotifyPipelineStatusError(
            f"Missing or invalid required field: {field_name}"
        )

    return value.strip()


def validate_status(status: str) -> str:
    """Validate and normalize notification status."""
    normalized_status = status.strip().lower()
    if normalized_status not in VALID_STATUSES:
        raise NotifyPipelineStatusError(
            f"Unsupported status={status}. Supported={sorted(VALID_STATUSES)}"
        )

    return normalized_status


def build_notification_message(event: dict[str, Any]) -> dict[str, Any]:
    """Build a structured notification payload."""
    status = validate_status(require_string_field(event, "status"))
    dataset_mode = str(event.get("dataset_mode", "unknown"))
    run_id = str(event.get("run_id", "unknown"))
    s3_bucket = str(event.get("s3_bucket", "unknown"))
    s3_prefix = normalize_prefix(event.get("s3_prefix"))
    error = event.get("error")

    evaluation_summary_uri = event.get("evaluation_summary_uri")
    model_name = event.get("model_name")
    best_f1_score = event.get("best_f1_score")
    best_roc_auc_score = event.get("best_roc_auc_score")

    return {
        "service": "ColliderOpsAI",
        "status": status,
        "dataset_mode": dataset_mode,
        "run_id": run_id,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "model_name": model_name,
        "best_f1_score": best_f1_score,
        "best_roc_auc_score": best_roc_auc_score,
        "evaluation_summary_uri": evaluation_summary_uri,
        "error": error,
        "notified_at": datetime.now(timezone.utc).isoformat(),
    }


def build_subject(message: dict[str, Any]) -> str:
    """Build a compact notification subject."""
    status = message["status"]
    dataset_mode = message.get("dataset_mode", "unknown")
    return f"ColliderOpsAI pipeline {status}: {dataset_mode}"


def build_notification_key(message: dict[str, Any]) -> str:
    """Build the S3 key for a notification audit record."""
    s3_prefix = normalize_prefix(str(message.get("s3_prefix", DEFAULT_S3_PREFIX)))
    dataset_mode = str(message.get("dataset_mode", "unknown")).strip() or "unknown"
    run_id = str(message.get("run_id", "unknown")).strip() or "unknown"
    status = str(message.get("status", "unknown")).strip() or "unknown"
    notified_at = str(message.get("notified_at", datetime.now(timezone.utc).isoformat()))
    safe_notified_at = quote_plus(notified_at)

    return (
        f"{s3_prefix}/notifications/{dataset_mode}/{run_id}/"
        f"{status}_{safe_notified_at}.json"
    )


def write_notification_to_s3_if_configured(message: dict[str, Any]) -> str | None:
    """Write notification payload to S3 when enabled.

    Local runs and Step Functions can use this as an audit trail before SNS is
    configured. Set COLLIDEROPS_WRITE_NOTIFICATION_TO_S3=true to enable it.
    """
    should_write = os.getenv(WRITE_NOTIFICATION_TO_S3_ENV_VAR, "false").strip().lower()
    if should_write not in {"1", "true", "yes"}:
        return None

    s3_bucket = str(message.get("s3_bucket", "")).strip()
    if not s3_bucket or s3_bucket == "unknown":
        raise NotifyPipelineStatusError(
            "Cannot write notification to S3 because s3_bucket is missing."
        )

    notification_key = build_notification_key(message=message)
    s3_client = boto3.client("s3")
    try:
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=notification_key,
            Body=json.dumps(message, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
    except ClientError as error:
        raise NotifyPipelineStatusError(
            f"Failed to write notification to s3://{s3_bucket}/{notification_key}: {error}"
        ) from error

    return f"s3://{s3_bucket}/{notification_key}"


def publish_to_sns_if_configured(message: dict[str, Any]) -> str | None:
    """Publish the notification to SNS when a topic ARN is configured."""
    topic_arn = os.getenv(SNS_TOPIC_ARN_ENV_VAR)
    if not topic_arn:
        return None

    sns_client = boto3.client("sns")
    try:
        response = sns_client.publish(
            TopicArn=topic_arn,
            Subject=build_subject(message),
            Message=json.dumps(message, indent=2),
        )
    except ClientError as error:
        raise NotifyPipelineStatusError(
            f"Failed to publish SNS notification: {error}"
        ) from error

    return response.get("MessageId")


def notify_pipeline_status(event: dict[str, Any]) -> dict[str, Any]:
    """Notify pipeline status and return the notification payload."""
    message = build_notification_message(event=event)

    # CloudWatch captures stdout from Lambda. This gives us a default audit trail
    # even when SNS is not configured yet.
    print(json.dumps(message, indent=2))

    notification_s3_uri = write_notification_to_s3_if_configured(message=message)
    sns_message_id = publish_to_sns_if_configured(message=message)

    return {
        "notified": True,
        "notification": message,
        "notification_s3_uri": notification_s3_uri,
        "sns_message_id": sns_message_id,
    }


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda entrypoint."""
    try:
        payload = notify_pipeline_status(event=event)
        return build_response(status_code=200, body=payload)
    except NotifyPipelineStatusError as error:
        return build_response(
            status_code=400,
            body={
                "notified": False,
                "error_type": "NotifyPipelineStatusError",
                "message": str(error),
            },
        )
    except Exception as error:
        return build_response(
            status_code=500,
            body={
                "notified": False,
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )


if __name__ == "__main__":
    sample_event = {
        "status": "success",
        "dataset_mode": "curated_higgs",
        "run_id": "curated_higgs-local-test",
        "s3_bucket": "colliderops-ai-dev-ap-southeast-2",
        "s3_prefix": "dev",
        "model_name": "hist_gradient_boosting_curated_higgs",
        "best_f1_score": 0.700,
        "best_roc_auc_score": 0.786,
        "evaluation_summary_uri": "s3://colliderops-ai-dev-ap-southeast-2/dev/evaluation/curated_higgs/curated_higgs-local-test/evaluation_summary.md",
    }
    print(json.dumps(lambda_handler(sample_event, None), indent=2))