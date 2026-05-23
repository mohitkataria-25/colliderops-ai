

"""Real CERN/Open Data ETL job for ColliderOpsAI.

This script is the first ETL bridge from real CERN Open Data ROOT files into the
ColliderOpsAI data pipeline.

Current v1 scope:
- Read one registered CERN Open Data record from data/dataset_registry.json.
- Extract readable event-level features from selected ROOT files using root_adapter.
- Write processed event rows to CSV.
- Write curated ML-ready rows to CSV.

Current limitation:
- This version supports a signal-only sample from record 7901 by default.
- A background record should be added later to create a proper binary classifier
  dataset from real CERN/Open Data.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from etl.adapters.root_adapter import extract_registered_root_event_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_OUTPUT_DIR = DATA_DIR / "processed" / "real_cern_events"
CURATED_OUTPUT_DIR = DATA_DIR / "curated" / "real_cern_training_dataset"

PROCESSED_OUTPUT_PATH = PROCESSED_OUTPUT_DIR / "events.csv"
CURATED_OUTPUT_PATH = CURATED_OUTPUT_DIR / "training_dataset.csv"
RUN_METADATA_PATH = PROCESSED_OUTPUT_DIR / "run_metadata.json"

DEFAULT_RECORD_ID = "7901"
DEFAULT_LABEL = "signal"
DEFAULT_FILE_INDEXES = [0]
DEFAULT_MAX_EVENTS_PER_FILE = 100

CURATED_FEATURE_COLUMNS = [
    "gen_event_present",
    "gen_event_weight_count",
    "gen_event_signal_process_id",
    "gen_event_qscale",
    "gen_particles_present",
    "gen_particle_count",
    "ak5_genjets_present",
]

LABEL_COLUMN = "label"


def extract_real_cern_rows(
    record_id: str = DEFAULT_RECORD_ID,
    label: str = DEFAULT_LABEL,
    file_indexes: list[int] | None = None,
    max_events_per_file: int = DEFAULT_MAX_EVENTS_PER_FILE,
) -> list[dict[str, Any]]:
    """Extract readable event feature rows from registered CERN ROOT files."""
    selected_file_indexes = file_indexes or DEFAULT_FILE_INDEXES
    all_rows: list[dict[str, Any]] = []

    for file_index in selected_file_indexes:
        rows = extract_registered_root_event_features(
            record_id=record_id,
            file_index=file_index,
            label=label,
            max_events=max_events_per_file,
        )

        for row in rows:
            row["source_file_index"] = file_index

        all_rows.extend(rows)

    return all_rows


def build_curated_rows(processed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build ML-ready curated rows from processed CERN event rows."""
    curated_rows: list[dict[str, Any]] = []

    for row in processed_rows:
        curated_row: dict[str, Any] = {}

        for column in CURATED_FEATURE_COLUMNS:
            curated_row[column] = row.get(column)

        curated_row[LABEL_COLUMN] = row.get(LABEL_COLUMN)
        curated_rows.append(curated_row)

    return curated_rows


def write_rows_to_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    """Write rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"No rows available to write to {output_path}.")

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def write_run_metadata(
    processed_rows: list[dict[str, Any]],
    curated_rows: list[dict[str, Any]],
    record_id: str,
    label: str,
    file_indexes: list[int],
    max_events_per_file: int,
) -> Path:
    """Write ETL run metadata for reproducibility."""
    RUN_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "job_name": "real_cern_etl_job",
        "status": "success",
        "record_id": str(record_id),
        "label": label,
        "file_indexes": file_indexes,
        "max_events_per_file": max_events_per_file,
        "processed_output_path": str(PROCESSED_OUTPUT_PATH),
        "curated_output_path": str(CURATED_OUTPUT_PATH),
        "processed_row_count": len(processed_rows),
        "curated_row_count": len(curated_rows),
        "curated_feature_columns": CURATED_FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "This ETL job extracts a small readable subset from real CERN/CMS ROOT files.",
            "Current v1 output is signal-only by default because a matching background record has not yet been wired.",
            "The curated dataset is suitable for pipeline testing, but not yet for meaningful binary classification training.",
        ],
    }

    with RUN_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return RUN_METADATA_PATH


def run_real_cern_etl(
    record_id: str = DEFAULT_RECORD_ID,
    label: str = DEFAULT_LABEL,
    file_indexes: list[int] | None = None,
    max_events_per_file: int = DEFAULT_MAX_EVENTS_PER_FILE,
) -> dict[str, Any]:
    """Run the real CERN/Open Data ETL job."""
    selected_file_indexes = file_indexes or DEFAULT_FILE_INDEXES

    processed_rows = extract_real_cern_rows(
        record_id=record_id,
        label=label,
        file_indexes=selected_file_indexes,
        max_events_per_file=max_events_per_file,
    )
    curated_rows = build_curated_rows(processed_rows=processed_rows)

    processed_path = write_rows_to_csv(
        rows=processed_rows,
        output_path=PROCESSED_OUTPUT_PATH,
    )
    curated_path = write_rows_to_csv(
        rows=curated_rows,
        output_path=CURATED_OUTPUT_PATH,
    )
    metadata_path = write_run_metadata(
        processed_rows=processed_rows,
        curated_rows=curated_rows,
        record_id=record_id,
        label=label,
        file_indexes=selected_file_indexes,
        max_events_per_file=max_events_per_file,
    )

    return {
        "job_name": "real_cern_etl_job",
        "status": "success",
        "record_id": str(record_id),
        "label": label,
        "file_indexes": selected_file_indexes,
        "max_events_per_file": max_events_per_file,
        "processed_row_count": len(processed_rows),
        "curated_row_count": len(curated_rows),
        "processed_output_path": str(processed_path),
        "curated_output_path": str(curated_path),
        "run_metadata_path": str(metadata_path),
    }


def print_etl_summary(summary: dict[str, Any]) -> None:
    """Print ETL job summary."""
    print("-" * 80)
    print(f"Job: {summary.get('job_name')}")
    print(f"Status: {summary.get('status')}")
    print(f"Record ID: {summary.get('record_id')}")
    print(f"Label: {summary.get('label')}")
    print(f"File indexes: {summary.get('file_indexes')}")
    print(f"Max events per file: {summary.get('max_events_per_file')}")
    print(f"Processed rows: {summary.get('processed_row_count')}")
    print(f"Curated rows: {summary.get('curated_row_count')}")
    print(f"Processed output: {summary.get('processed_output_path')}")
    print(f"Curated output: {summary.get('curated_output_path')}")
    print(f"Run metadata: {summary.get('run_metadata_path')}")


def main() -> None:
    """Run the real CERN/Open Data ETL job with default settings."""
    summary = run_real_cern_etl()
    print_etl_summary(summary=summary)


if __name__ == "__main__":
    main()