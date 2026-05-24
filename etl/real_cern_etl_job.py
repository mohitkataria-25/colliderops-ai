"""Real CERN/Open Data ETL job for ColliderOpsAI.

This script is the first ETL bridge from real CERN Open Data ROOT files into the
ColliderOpsAI data pipeline.

Current v2 scope:
- Combines one signal sample and one background sample by default.
- Signal default: CMS Higgs-to-gamma-gamma Monte Carlo record 7901.
- Background default: CMS GJets Monte Carlo record 7779.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from etl.adapters.root_adapter import (
    extract_readable_event_features,
    get_root_file_url,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_OUTPUT_DIR = DATA_DIR / "processed" / "real_cern_events"
RAW_ROOT_CACHE_DIR = DATA_DIR / "raw" / "cern_root"
CURATED_OUTPUT_DIR = DATA_DIR / "curated" / "real_cern_training_dataset"

PROCESSED_OUTPUT_PATH = PROCESSED_OUTPUT_DIR / "events.csv"
CURATED_OUTPUT_PATH = CURATED_OUTPUT_DIR / "training_dataset.csv"
RUN_METADATA_PATH = PROCESSED_OUTPUT_DIR / "run_metadata.json"

DEFAULT_DATASET_CONFIGS = [
    {
        "record_id": "7901",
        "label": "signal",
        "file_indexes": [0, 1],
    },
    {
        "record_id": "7779",
        "label": "background",
        "file_indexes": [0, 1],
    },
]
DEFAULT_MAX_EVENTS_PER_FILE = 1000
FALLBACK_MAX_EVENTS_PER_FILE = [1000, 500, 100]
REMOTE_READ_RETRY_SLEEP_SECONDS = 10

CURATED_FEATURE_COLUMNS = [
    "gen_event_present",
    "gen_event_weight_count",
    "gen_event_weight_min",
    "gen_event_weight_max",
    "gen_event_weight_mean",
    "gen_event_weight_std",
    "gen_event_weight_sum",
    "gen_event_weight_unique_count",
    "gen_event_qscale",
    "gen_particles_present",
    "gen_particle_count",
    "gen_particle_id_min",
    "gen_particle_id_max",
    "gen_particle_id_mean",
    "gen_particle_id_std",
    "gen_particle_id_sum",
    "gen_particle_id_unique_count",
    "ak5_genjets_present",
]


LABEL_COLUMN = "label"


def build_cached_root_file_path(record_id: str, file_index: int, file_url: str) -> Path:
    """Build a stable local cache path for a CERN ROOT file."""
    file_name = file_url.rstrip("/").split("/")[-1]
    safe_file_name = f"file_{file_index}_{file_name}"
    return RAW_ROOT_CACHE_DIR / str(record_id) / safe_file_name


def get_or_download_root_file(record_id: str, file_index: int) -> Path:
    """Download a CERN ROOT file once and reuse the local cached copy.

    Streaming remote ROOT files repeatedly with uproot can trigger CERN Open Data
    HTTP range-request failures or 429 rate limits. Local caching makes scaled
    experimentation more stable and reproducible.
    """
    import urllib.request

    file_url = get_root_file_url(record_id=record_id, file_index=file_index)
    local_path = build_cached_root_file_path(
        record_id=record_id,
        file_index=file_index,
        file_url=file_url,
    )

    if local_path.exists() and local_path.stat().st_size > 0:
        print(f"Using cached ROOT file: {local_path}")
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ROOT file for local cache: {file_url}")
    print(f"Cache destination: {local_path}")
    time.sleep(REMOTE_READ_RETRY_SLEEP_SECONDS)

    try:
        urllib.request.urlretrieve(file_url, local_path)
    except Exception:
        if local_path.exists():
            local_path.unlink()
        raise

    return local_path


def extract_file_rows_with_fallback(
    record_id: str,
    label: str,
    file_index: int,
    max_events_per_file: int,
) -> list[dict[str, Any]]:
    """Extract rows from one ROOT file, retrying with smaller event windows if needed.

    Remote CERN ROOT reads can occasionally fail because of network/range-request
    issues. For scaled local experimentation, this keeps the ETL resilient by
    retrying a smaller number of events from the same file before giving up.
    """
    attempted_event_limits = [
        max_events_per_file,
        *FALLBACK_MAX_EVENTS_PER_FILE,
    ]
    attempted_event_limits = list(dict.fromkeys(attempted_event_limits))

    last_error: Exception | None = None

    try:
        local_root_file_path = get_or_download_root_file(
            record_id=record_id,
            file_index=file_index,
        )
    except Exception as download_error:
        print(
            "Warning: skipping ROOT file because local cache download failed. "
            f"record_id={record_id}, label={label}, file_index={file_index}, "
            f"error={download_error}"
        )
        return []

    for event_limit in attempted_event_limits:
        try:
            rows = extract_readable_event_features(
                file_path_or_url=str(local_root_file_path),
                record_id=record_id,
                label=label,
                max_events=event_limit,
            )

            for row in rows:
                row["source_file_index"] = file_index
                row["source_requested_events_per_file"] = max_events_per_file
                row["source_extracted_events_limit"] = event_limit
                row["source_local_root_file_path"] = str(local_root_file_path)

            return rows

        except Exception as error:
            last_error = error
            print(
                "Warning: ROOT extraction failed "
                f"for record_id={record_id}, label={label}, "
                f"file_index={file_index}, event_limit={event_limit}. "
                f"Error: {error}"
            )
            time.sleep(REMOTE_READ_RETRY_SLEEP_SECONDS)

    print(
        "Warning: skipping ROOT file after all fallback attempts failed. "
        f"record_id={record_id}, label={label}, file_index={file_index}, "
        f"last_error={last_error}"
    )
    return []


def extract_real_cern_rows(
    record_id: str,
    label: str,
    file_indexes: list[int],
    max_events_per_file: int = DEFAULT_MAX_EVENTS_PER_FILE,
) -> list[dict[str, Any]]:
    """Extract readable event feature rows from registered CERN ROOT files."""

    all_rows: list[dict[str, Any]] = []

    for file_index in file_indexes:
        rows = extract_file_rows_with_fallback(
            record_id=record_id,
            label=label,
            file_index=file_index,
            max_events_per_file=max_events_per_file,
        )

        all_rows.extend(rows)
        print(
            "Extraction progress: "
            f"record_id={record_id}, label={label}, file_index={file_index}, "
            f"rows_extracted={len(rows)}, cumulative_rows={len(all_rows)}"
        )

    return all_rows


def extract_real_cern_rows_from_configs(
    dataset_configs: list[dict[str, Any]] | None = None,
    max_events_per_file: int = DEFAULT_MAX_EVENTS_PER_FILE,
) -> list[dict[str, Any]]:
    """Extract readable event feature rows from multiple CERN dataset configs."""
    selected_configs = dataset_configs or DEFAULT_DATASET_CONFIGS
    all_rows: list[dict[str, Any]] = []

    for config in selected_configs:
        record_id = str(config["record_id"])
        label = str(config["label"])
        file_indexes = config.get("file_indexes", [0])

        rows = extract_real_cern_rows(
            record_id=record_id,
            label=label,
            file_indexes=file_indexes,
            max_events_per_file=max_events_per_file,
        )

        for row in rows:
            row["dataset_label"] = label

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
    dataset_configs: list[dict[str, Any]],
    max_events_per_file: int,
) -> Path:
    """Write ETL run metadata for reproducibility."""
    RUN_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    label_counts: dict[str, int] = {}
    for row in curated_rows:
        label = str(row.get(LABEL_COLUMN))
        label_counts[label] = label_counts.get(label, 0) + 1

    metadata = {
        "job_name": "real_cern_etl_job",
        "status": "success",
        "dataset_configs": dataset_configs,
        "label_counts": label_counts,
        "two_class_training_ready": len(label_counts) >= 2,
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
            "Default v2 output combines signal record 7901 and background record 7779.",
            "This is a first real CERN/Open Data binary dataset and should still be treated as a prototype feature set.",
            "Scaled ETL uses per-file fallback extraction to handle intermittent remote ROOT read failures.",
            "Scaled ETL caches CERN ROOT files locally under data/raw/cern_root before extraction.",
        ],
    }

    with RUN_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return RUN_METADATA_PATH


def run_real_cern_etl(
    dataset_configs: list[dict[str, Any]] | None = None,
    max_events_per_file: int = DEFAULT_MAX_EVENTS_PER_FILE,
) -> dict[str, Any]:
    """Run the real CERN/Open Data ETL job."""
    selected_configs = dataset_configs or DEFAULT_DATASET_CONFIGS

    processed_rows = extract_real_cern_rows_from_configs(
        dataset_configs=selected_configs,
        max_events_per_file=max_events_per_file,
    )
    if not processed_rows:
        raise RuntimeError(
            "Real CERN ETL extracted zero rows. This is usually caused by remote "
            "CERN Open Data rate limiting during local ROOT file download, or unreadable "
            "ROOT files. Wait a few minutes, reduce DEFAULT_MAX_EVENTS_PER_FILE, use "
            "fewer file indexes, or rerun after cached ROOT files are available. Existing "
            "curated data was not overwritten."
        )
    curated_rows = build_curated_rows(processed_rows=processed_rows)

    label_counts: dict[str, int] = {}
    for row in curated_rows:
        label = str(row.get(LABEL_COLUMN))
        label_counts[label] = label_counts.get(label, 0) + 1

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
        dataset_configs=selected_configs,
        max_events_per_file=max_events_per_file,
    )

    return {
        "job_name": "real_cern_etl_job",
        "status": "success",
        "dataset_configs": selected_configs,
        "label_counts": label_counts,
        "two_class_training_ready": len(label_counts) >= 2,
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
    print(f"Dataset configs: {summary.get('dataset_configs')}")
    print(f"Label counts: {summary.get('label_counts')}")
    print(f"Two-class training ready: {summary.get('two_class_training_ready')}")
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