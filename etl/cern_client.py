"""CERN Open Data client utilities for ColliderOpsAI.

This module wraps the official `cernopendata-client` CLI for the first real-world
CERN/Open Data ingestion path.

Current scope:
- Fetch record metadata by CERN Open Data record ID.
- Fetch file locations for a record.
- Normalize record metadata into a local summary.
- Save source metadata into data/dataset_registry.json.

This module does not transform physics data yet. Dataset transformation belongs in
future adapter modules such as csv_adapter.py, parquet_adapter.py, and root_adapter.py.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_REGISTRY_PATH = DATA_DIR / "dataset_registry.json"

DEFAULT_RECORD_ID = "5500"
DEFAULT_PROTOCOL = "http"


def ensure_cern_client_available() -> None:
    """Raise an error if cernopendata-client is not available in the environment."""
    if shutil.which("cernopendata-client") is None:
        raise RuntimeError(
            "cernopendata-client is not installed or not available on PATH. "
            "Install it with: pip install cernopendata-client"
        )


def run_cern_client_command(args: list[str]) -> str:
    """Run a cernopendata-client command and return stdout."""
    ensure_cern_client_available()

    command = ["cernopendata-client", *args]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def fetch_record_metadata(record_id: str) -> dict[str, Any]:
    """Fetch CERN Open Data metadata for a record ID."""
    output = run_cern_client_command(
        ["get-metadata", "--recid", str(record_id)]
    )

    try:
        return json.loads(output)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Could not parse metadata JSON for CERN record {record_id}."
        ) from error


def fetch_file_locations(
    record_id: str,
    protocol: str = DEFAULT_PROTOCOL,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Fetch file locations for a CERN Open Data record."""
    args = [
        "get-file-locations",
        "--recid",
        str(record_id),
        "--protocol",
        protocol,
    ]

    if verbose:
        args.append("--verbose")

    output = run_cern_client_command(args)

    if not output:
        return []

    return [
        _parse_file_location_line(line=line, verbose=verbose)
        for line in output.splitlines()
    ]


def _parse_file_location_line(line: str, verbose: bool = False) -> dict[str, Any]:
    """Parse one get-file-locations output line into a structured dictionary."""
    parts = line.strip().split()
    file_url = parts[0]

    parsed_url = urlparse(file_url)
    file_name = Path(parsed_url.path).name
    file_format = _infer_file_format(file_name=file_name)

    file_info: dict[str, Any] = {
        "file_name": file_name,
        "file_url": file_url,
        "file_format": file_format,
        "file_size": None,
        "checksum": None,
    }

    if verbose and len(parts) >= 2:
        file_info["file_size"] = parts[1]

    if verbose and len(parts) >= 3:
        file_info["checksum"] = parts[2]

    return file_info


def _infer_file_format(file_name: str) -> str:
    """Infer a lightweight file format label from a file name."""
    suffix = Path(file_name).suffix.lower().replace(".", "")

    if not suffix:
        return "unknown"

    return suffix


def extract_record_summary(
    record_id: str,
    metadata: dict[str, Any],
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    """Normalize CERN metadata and file locations into a local dataset summary."""
    return {
        "dataset_name": _safe_get(metadata, "title"),
        "cern_record_id": str(record_id),
        "source": "CERN Open Data Portal",
        "source_url": f"https://opendata.cern.ch/record/{record_id}",
        "doi": _safe_get(metadata, "doi"),
        "license": _extract_license(metadata),
        "experiment": _extract_experiment(metadata),
        "description": _safe_get(metadata, "description"),
        "keywords": _safe_get(metadata, "keywords", default=[]),
        "publication_date": _safe_get(metadata, "publication_date"),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "files": files,
        "ingestion_status": "metadata_registered",
        "notes": [
            "This registry entry captures CERN Open Data source metadata and file locations.",
            "Physics data transformation is not performed by cern_client.py.",
            "Use adapter modules later to convert selected files into processed and curated datasets.",
        ],
    }


def _safe_get(metadata: dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely read a key from metadata."""
    value = metadata.get(key, default)
    return default if value is None else value


def _extract_license(metadata: dict[str, Any]) -> Any:
    """Extract license information from record metadata when available."""
    license_info = metadata.get("license") or metadata.get("licenses")

    if isinstance(license_info, list):
        return license_info

    if isinstance(license_info, dict):
        return license_info

    return license_info


def _extract_experiment(metadata: dict[str, Any]) -> Any:
    """Extract experiment/collaboration information from metadata when available."""
    accelerator_experiment = metadata.get("accelerator_experiment")

    if accelerator_experiment:
        return accelerator_experiment

    keywords = metadata.get("keywords", [])

    if isinstance(keywords, list):
        for keyword in keywords:
            normalized_keyword = str(keyword).lower()
            if normalized_keyword in {"cms", "atlas", "alice", "lhcb"}:
                return str(keyword).upper()

    return None


def load_dataset_registry() -> list[dict[str, Any]]:
    """Load the local dataset registry."""
    if not DATASET_REGISTRY_PATH.exists():
        return []

    with DATASET_REGISTRY_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    return [data]


def save_dataset_registry(record_summary: dict[str, Any]) -> Path:
    """Save or update a CERN record summary in the local dataset registry."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    registry = load_dataset_registry()
    record_id = record_summary["cern_record_id"]

    updated_registry = [
        record for record in registry if record.get("cern_record_id") != record_id
    ]
    updated_registry.append(record_summary)

    with DATASET_REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump(updated_registry, file, indent=2)

    return DATASET_REGISTRY_PATH


def register_cern_record(
    record_id: str = DEFAULT_RECORD_ID,
    protocol: str = DEFAULT_PROTOCOL,
    verbose_files: bool = False,
) -> dict[str, Any]:
    """Fetch metadata/file locations for one CERN record and save it to the registry."""
    metadata = fetch_record_metadata(record_id=record_id)
    files = fetch_file_locations(
        record_id=record_id,
        protocol=protocol,
        verbose=verbose_files,
    )

    record_summary = extract_record_summary(
        record_id=record_id,
        metadata=metadata,
        files=files,
    )

    registry_path = save_dataset_registry(record_summary=record_summary)
    record_summary["registry_path"] = str(registry_path)

    return record_summary


def main() -> None:
    """Register a default CERN Open Data record as a smoke test."""
    summary = register_cern_record(record_id=DEFAULT_RECORD_ID)

    print(f"Registered CERN Open Data record: {summary['cern_record_id']}")
    print(f"Dataset name: {summary.get('dataset_name')}")
    print(f"File count: {summary.get('file_count')}")
    print(f"Registry path: {summary.get('registry_path')}")


if __name__ == "__main__":
    main()