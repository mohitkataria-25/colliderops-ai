

"""Download and curate an ML-ready HIGGS benchmark dataset sample.

This script adds a second dataset path to ColliderOpsAI:

1. `real_cern` remains the raw CERN/Open Data ROOT ingestion path.
2. `curated_higgs` becomes an analysis-ready ML benchmark path.

The default source is the public HIGGS dataset from the UCI Machine Learning
Repository. The dataset is a particle-physics signal/background benchmark where
column 0 is the class label and the remaining columns are physics-inspired event
features.

The script streams the compressed CSV in chunks, builds a balanced sample, and
writes both raw-sample and curated ML-ready CSV outputs.
"""

from __future__ import annotations

import argparse
import ssl
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

RAW_OUTPUT_DIR = DATA_DIR / "raw" / "curated_higgs"
CURATED_OUTPUT_DIR = DATA_DIR / "curated" / "curated_higgs_training_dataset"
RAW_OUTPUT_PATH = RAW_OUTPUT_DIR / "higgs_sample.csv"
SOURCE_CACHE_PATH = RAW_OUTPUT_DIR / "HIGGS.csv.gz"
CURATED_OUTPUT_PATH = CURATED_OUTPUT_DIR / "training_dataset.csv"
DATASET_CARD_PATH = CURATED_OUTPUT_DIR / "dataset_card.md"

DEFAULT_SOURCE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
)
DEFAULT_TOTAL_ROWS = 100_000
DEFAULT_CHUNK_SIZE = 50_000
LABEL_COLUMN = "label"

# HIGGS dataset layout:
# column 0 = label; columns 1-21 = low-level features; columns 22-28 = high-level features.
# These names follow the commonly documented HIGGS benchmark structure.
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

RAW_COLUMN_NAMES = [LABEL_COLUMN, *HIGGS_FEATURE_COLUMNS]
LABEL_MAPPING = {
    0: "background",
    1: "signal",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-url",
        default=DEFAULT_SOURCE_URL,
        help="CSV or CSV.GZ URL/path for the HIGGS benchmark dataset.",
    )
    parser.add_argument(
        "--total-rows",
        type=int,
        default=DEFAULT_TOTAL_ROWS,
        help="Total balanced rows to write across signal and background.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of rows to stream per pandas chunk.",
    )

    return parser.parse_args()


def validate_requested_rows(total_rows: int) -> None:
    """Validate total row request."""
    if total_rows < 2:
        raise ValueError("total_rows must be at least 2.")

    if total_rows % 2 != 0:
        raise ValueError("total_rows must be an even number for a balanced dataset.")


# -----------------------------------------------------------------------------
# Source file caching utility for robust downloads
# -----------------------------------------------------------------------------
def cache_source_file(source_url: str) -> str:
    """Download the HIGGS source file once and return a local path.

    Some Python/macOS environments fail SSL certificate verification against the
    UCI archive even when the URL is reachable from a browser. To keep the ETL
    usable for local experimentation, this function first tries normal SSL
    verification and then falls back to an unverified SSL context only when the
    failure is certificate-related.
    """
    if not source_url.startswith(("http://", "https://")):
        return source_url

    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if SOURCE_CACHE_PATH.exists() and SOURCE_CACHE_PATH.stat().st_size > 0:
        print(f"Using cached HIGGS source file: {SOURCE_CACHE_PATH}")
        return str(SOURCE_CACHE_PATH)

    print(f"Downloading HIGGS source file: {source_url}")
    print(f"Cache destination: {SOURCE_CACHE_PATH}")

    try:
        urllib.request.urlretrieve(source_url, SOURCE_CACHE_PATH)
    except Exception as error:
        if "CERTIFICATE_VERIFY_FAILED" not in str(error):
            if SOURCE_CACHE_PATH.exists():
                SOURCE_CACHE_PATH.unlink()
            raise

        print(
            "Warning: SSL certificate verification failed for the source URL. "
            "Retrying download with an unverified SSL context for local dataset caching."
        )
        context = ssl._create_unverified_context()
        request = urllib.request.Request(source_url)
        with urllib.request.urlopen(request, context=context) as response:
            SOURCE_CACHE_PATH.write_bytes(response.read())

    if not SOURCE_CACHE_PATH.exists() or SOURCE_CACHE_PATH.stat().st_size == 0:
        raise RuntimeError(f"Failed to cache source file at {SOURCE_CACHE_PATH}")

    return str(SOURCE_CACHE_PATH)


def normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Map numeric labels to readable signal/background labels."""
    normalized_df = df.copy()
    normalized_df[LABEL_COLUMN] = normalized_df[LABEL_COLUMN].astype(int).map(
        LABEL_MAPPING
    )

    if normalized_df[LABEL_COLUMN].isnull().any():
        invalid_values = df.loc[normalized_df[LABEL_COLUMN].isnull(), LABEL_COLUMN].unique()
        raise ValueError(f"Invalid HIGGS labels found: {invalid_values}")

    return normalized_df


def select_balanced_rows(
    source_url: str,
    total_rows: int = DEFAULT_TOTAL_ROWS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> pd.DataFrame:
    """Stream source data and return a balanced signal/background sample."""
    validate_requested_rows(total_rows=total_rows)

    rows_per_label = total_rows // 2
    collected_frames: list[pd.DataFrame] = []
    label_counts = {
        "signal": 0,
        "background": 0,
    }

    source_path = cache_source_file(source_url=source_url)
    reader = pd.read_csv(
        source_path,
        compression="infer",
        header=None,
        names=RAW_COLUMN_NAMES,
        chunksize=chunk_size,
    )

    for chunk in reader:
        normalized_chunk = normalize_label_column(df=chunk)

        selected_chunks = []
        for label in ["signal", "background"]:
            remaining = rows_per_label - label_counts[label]
            if remaining <= 0:
                continue

            label_rows = normalized_chunk[normalized_chunk[LABEL_COLUMN] == label].head(
                remaining
            )
            if not label_rows.empty:
                selected_chunks.append(label_rows)
                label_counts[label] += len(label_rows)

        if selected_chunks:
            collected_frames.append(pd.concat(selected_chunks, ignore_index=True))

        if all(count >= rows_per_label for count in label_counts.values()):
            break

    if not collected_frames:
        raise ValueError("No rows were collected from the HIGGS source dataset.")

    balanced_df = pd.concat(collected_frames, ignore_index=True)

    if len(balanced_df) < total_rows:
        raise ValueError(
            "Unable to collect the requested balanced row count. "
            f"Requested={total_rows}, collected={len(balanced_df)}, "
            f"label_counts={label_counts}"
        )

    balanced_df = balanced_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return balanced_df


def validate_curated_higgs_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Validate curated HIGGS dataframe shape and feature readiness."""
    required_columns = [LABEL_COLUMN, *HIGGS_FEATURE_COLUMNS]
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    feature_df = df[HIGGS_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    null_feature_count = int(feature_df.isnull().sum().sum())
    if null_feature_count > 0:
        raise ValueError(
            f"Curated HIGGS features contain null/non-numeric values: {null_feature_count}"
        )

    label_counts = df[LABEL_COLUMN].value_counts().to_dict()
    unique_labels = sorted(df[LABEL_COLUMN].unique().tolist())

    return {
        "dataset_mode": "curated_higgs",
        "row_count": len(df),
        "feature_count": len(HIGGS_FEATURE_COLUMNS),
        "feature_columns": HIGGS_FEATURE_COLUMNS,
        "label_counts": label_counts,
        "unique_labels": unique_labels,
        "two_class_training_ready": len(unique_labels) == 2,
        "schema_valid": True,
        "features_numeric": True,
        "null_feature_count": null_feature_count,
    }


def write_dataset_card(
    source_url: str,
    total_rows: int,
    validation_summary: dict[str, Any],
) -> Path:
    """Write a compact dataset card for the curated HIGGS sample."""
    DATASET_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)

    card = f"""# Curated HIGGS Dataset Card

## Purpose

This dataset mode provides an ML-ready particle-physics benchmark dataset for ColliderOpsAI.
It complements the `real_cern` ROOT ingestion path by adding a curated feature table that is easier to model and benchmark.

## Source

Source URL/path:

```text
{source_url}
```

## Output

Raw sample:

```text
{RAW_OUTPUT_PATH}
```

Curated training dataset:

```text
{CURATED_OUTPUT_PATH}
```

## Shape

Requested rows: {total_rows}
Actual rows: {validation_summary.get('row_count')}
Feature count: {validation_summary.get('feature_count')}
Label counts: {validation_summary.get('label_counts')}

## Notes

- Label `signal` maps from source label `1`.
- Label `background` maps from source label `0`.
- This curated benchmark should not be confused with the raw CERN ROOT ingestion pipeline.
- Use this dataset for ML benchmarking and model comparison.
- Use `real_cern` mode for raw ROOT ingestion, caching, and feature extraction workflows.
"""

    DATASET_CARD_PATH.write_text(card)
    return DATASET_CARD_PATH


def write_outputs(
    df: pd.DataFrame,
    source_url: str,
    total_rows: int,
) -> dict[str, Any]:
    """Write raw sample, curated sample, and dataset card outputs."""
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CURATED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(RAW_OUTPUT_PATH, index=False)
    df[[*HIGGS_FEATURE_COLUMNS, LABEL_COLUMN]].to_csv(
        CURATED_OUTPUT_PATH,
        index=False,
    )

    validation_summary = validate_curated_higgs_dataset(df=df)
    dataset_card_path = write_dataset_card(
        source_url=source_url,
        total_rows=total_rows,
        validation_summary=validation_summary,
    )

    return {
        "job_name": "download_curated_higgs_dataset",
        "status": "success",
        "source_url": source_url,
        "raw_output_path": str(RAW_OUTPUT_PATH),
        "curated_output_path": str(CURATED_OUTPUT_PATH),
        "dataset_card_path": str(dataset_card_path),
        **validation_summary,
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Print a compact job summary."""
    print("-" * 80)
    print(f"Job: {summary.get('job_name')}")
    print(f"Status: {summary.get('status')}")
    print(f"Source: {summary.get('source_url')}")
    print(f"Rows: {summary.get('row_count')}")
    print(f"Features: {summary.get('feature_count')}")
    print(f"Label counts: {summary.get('label_counts')}")
    print(f"Two-class training ready: {summary.get('two_class_training_ready')}")
    print(f"Raw output: {summary.get('raw_output_path')}")
    print(f"Curated output: {summary.get('curated_output_path')}")
    print(f"Dataset card: {summary.get('dataset_card_path')}")


def run_download_curated_higgs_dataset(
    source_url: str = DEFAULT_SOURCE_URL,
    total_rows: int = DEFAULT_TOTAL_ROWS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Run the curated HIGGS download/curation job."""
    df = select_balanced_rows(
        source_url=source_url,
        total_rows=total_rows,
        chunk_size=chunk_size,
    )
    return write_outputs(
        df=df,
        source_url=source_url,
        total_rows=total_rows,
    )


def main() -> None:
    args = parse_args()
    summary = run_download_curated_higgs_dataset(
        source_url=args.source_url,
        total_rows=args.total_rows,
        chunk_size=args.chunk_size,
    )
    print_summary(summary=summary)


if __name__ == "__main__":
    main()