

"""Locally testable helpers for the curated HIGGS Glue ETL job.

This module intentionally avoids importing `awsglue`, `pyspark`, or `boto3`.
It contains pure Python constants and helper functions that can be tested in a
normal local virtual environment.

The Glue runtime wrapper should live in `curated_higgs_glue_job.py` and import
these helpers where useful.
"""

from __future__ import annotations

from typing import Any


DATASET_MODE = "curated_higgs"
DEFAULT_S3_PREFIX = "dev"
DEFAULT_TOTAL_ROWS = 10_000

RAW_HIGGS_KEY = "raw/curated_higgs/HIGGS.csv.gz"
CURATED_HIGGS_RELATIVE_PATH = "curated/curated_higgs/training_dataset.parquet"
CURATED_METADATA_RELATIVE_PATH = "curated/curated_higgs/dataset_metadata.json"

HIGGS_FEATURE_COLUMNS = [
    "lepton_pT",
    "lepton_eta",
    "lepton_phi",
    "missing_energy_magnitude",
    "missing_energy_phi",
    "jet_1_pt",
    "jet_1_eta",
    "jet_1_phi",
    "jet_1_b_tag",
    "jet_2_pt",
    "jet_2_eta",
    "jet_2_phi",
    "jet_2_b_tag",
    "jet_3_pt",
    "jet_3_eta",
    "jet_3_phi",
    "jet_3_b_tag",
    "jet_4_pt",
    "jet_4_eta",
    "jet_4_phi",
    "jet_4_b_tag",
    "m_jj",
    "m_jjj",
    "m_lv",
    "m_jlv",
    "m_bb",
    "m_wbb",
    "m_wwbb",
]

RAW_COLUMNS = ["raw_label", *HIGGS_FEATURE_COLUMNS]
REQUIRED_LABELS = ["background", "signal"]


class CuratedHiggsTransformError(ValueError):
    """Raised when curated HIGGS transform helper validation fails."""


def normalize_prefix(s3_prefix: str | None) -> str:
    """Normalize S3 prefix without leading/trailing slashes."""
    if not s3_prefix:
        return DEFAULT_S3_PREFIX

    return s3_prefix.strip().strip("/") or DEFAULT_S3_PREFIX


def build_s3_uri(bucket: str, s3_prefix: str, relative_path: str) -> str:
    """Build an S3 URI from bucket, prefix, and relative path."""
    if not bucket or not bucket.strip():
        raise CuratedHiggsTransformError("bucket must be a non-empty string.")

    normalized_prefix = normalize_prefix(s3_prefix=s3_prefix)
    normalized_relative_path = relative_path.strip("/")
    if not normalized_relative_path:
        raise CuratedHiggsTransformError("relative_path must be a non-empty string.")

    return f"s3://{bucket.strip()}/{normalized_prefix}/{normalized_relative_path}"


def build_s3_key(s3_prefix: str, relative_path: str) -> str:
    """Build an S3 object key from prefix and relative path."""
    normalized_prefix = normalize_prefix(s3_prefix=s3_prefix)
    normalized_relative_path = relative_path.strip("/")
    if not normalized_relative_path:
        raise CuratedHiggsTransformError("relative_path must be a non-empty string.")

    return f"{normalized_prefix}/{normalized_relative_path}"


def parse_total_rows(total_rows_value: str | int | None) -> int:
    """Parse and validate total_rows."""
    if total_rows_value is None:
        return DEFAULT_TOTAL_ROWS

    try:
        total_rows = int(total_rows_value)
    except (TypeError, ValueError) as error:
        raise CuratedHiggsTransformError("total_rows must be an integer.") from error

    if total_rows < 2:
        raise CuratedHiggsTransformError("total_rows must be at least 2.")

    if total_rows % 2 != 0:
        raise CuratedHiggsTransformError(
            "total_rows must be even so the curated sample can be class-balanced."
        )

    return total_rows


def rows_per_label(total_rows: int) -> int:
    """Return expected rows per class for a balanced binary sample."""
    return parse_total_rows(total_rows) // 2


def build_curated_higgs_paths(
    bucket: str,
    s3_prefix: str,
) -> dict[str, str]:
    """Build all curated HIGGS S3 URIs and keys used by the Glue job."""
    raw_input_uri = build_s3_uri(
        bucket=bucket,
        s3_prefix=s3_prefix,
        relative_path=RAW_HIGGS_KEY,
    )
    curated_output_uri = build_s3_uri(
        bucket=bucket,
        s3_prefix=s3_prefix,
        relative_path=CURATED_HIGGS_RELATIVE_PATH,
    )
    curated_metadata_key = build_s3_key(
        s3_prefix=s3_prefix,
        relative_path=CURATED_METADATA_RELATIVE_PATH,
    )

    return {
        "raw_input_uri": raw_input_uri,
        "curated_output_uri": curated_output_uri,
        "curated_metadata_key": curated_metadata_key,
    }


def validate_dataset_mode(dataset_mode: str) -> str:
    """Validate and normalize dataset mode."""
    normalized_dataset_mode = dataset_mode.strip().lower()
    if normalized_dataset_mode != DATASET_MODE:
        raise CuratedHiggsTransformError(
            f"Unsupported dataset_mode={dataset_mode}. Expected {DATASET_MODE}."
        )

    return normalized_dataset_mode


def validate_label_counts(
    label_counts: dict[str, Any],
    expected_total_rows: int,
) -> None:
    """Validate balanced binary label counts."""
    expected_rows_per_label = rows_per_label(total_rows=expected_total_rows)
    missing_labels = sorted(set(REQUIRED_LABELS) - set(label_counts))
    if missing_labels:
        raise CuratedHiggsTransformError(f"Missing required labels: {missing_labels}")

    for label in REQUIRED_LABELS:
        try:
            count = int(label_counts[label])
        except (TypeError, ValueError) as error:
            raise CuratedHiggsTransformError(
                f"Label count for {label} must be an integer."
            ) from error

        if count != expected_rows_per_label:
            raise CuratedHiggsTransformError(
                f"Label count mismatch for {label}. "
                f"Expected={expected_rows_per_label}, received={count}."
            )


def build_dataset_metadata(
    row_count: int,
    label_counts: dict[str, Any],
    null_feature_count: int,
) -> dict[str, Any]:
    """Build standard curated HIGGS dataset metadata."""
    validate_label_counts(
        label_counts=label_counts,
        expected_total_rows=row_count,
    )

    if null_feature_count != 0:
        raise CuratedHiggsTransformError(
            f"null_feature_count must be 0. Received={null_feature_count}."
        )

    return {
        "dataset_mode": DATASET_MODE,
        "row_count": row_count,
        "feature_count": len(HIGGS_FEATURE_COLUMNS),
        "feature_columns": HIGGS_FEATURE_COLUMNS,
        "label_counts": {label: int(label_counts[label]) for label in REQUIRED_LABELS},
        "unique_labels": sorted(REQUIRED_LABELS),
        "two_class_training_ready": True,
        "schema_valid": True,
        "features_numeric": True,
        "null_feature_count": null_feature_count,
    }