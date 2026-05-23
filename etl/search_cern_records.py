

"""Search utilities for finding candidate CERN Open Data records.

This module helps identify practical CERN/Open Data records for the first real-world
ColliderOpsAI ingestion path.

Current scope:
- Search the CERN Open Data Portal records API.
- Normalize search results into compact candidate summaries.
- Rank candidates by whether they contain processable file formats.
- Print a shortlist for manual dataset selection.

This module does not download or transform data. Once a record is selected, use
etl.cern_client.register_cern_record() to save its metadata/file locations into
data/dataset_registry.json.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen


CERN_RECORDS_API_URL = "https://opendata.cern.ch/api/records/"

DEFAULT_QUERY = "CMS Higgs"
DEFAULT_RESULT_SIZE = 10
DEFAULT_PROTOCOL = "http"
ENRICH_FILE_LOCATIONS = True

PROCESSABLE_FILE_FORMATS = {
    "csv",
    "json",
    "parquet",
    "root",
    "txt",
}

PREFERRED_FILE_FORMATS = {
    "csv",
    "json",
    "parquet",
    "root",
}


def search_cern_records(
    query: str = DEFAULT_QUERY,
    size: int = DEFAULT_RESULT_SIZE,
    enrich_file_locations: bool = ENRICH_FILE_LOCATIONS,
) -> list[dict[str, Any]]:
    """Search CERN Open Data records and return normalized candidate summaries."""
    raw_response = _fetch_records_api_response(query=query, size=size)
    raw_records = _extract_raw_records(raw_response=raw_response)

    candidates = [normalize_record_candidate(raw_record) for raw_record in raw_records]

    if enrich_file_locations:
        candidates = [enrich_candidate_file_locations(candidate) for candidate in candidates]

    return sorted(
        candidates,
        key=lambda candidate: candidate.get("candidate_score", 0),
        reverse=True,
    )
def ensure_cern_client_available() -> bool:
    """Return True if cernopendata-client is available."""
    return shutil.which("cernopendata-client") is not None


def run_cern_client_command(args: list[str]) -> str:
    """Run a cernopendata-client command and return stdout."""
    if not ensure_cern_client_available():
        raise RuntimeError(
            "cernopendata-client is not installed or not available on PATH. "
            "Install it with: pip install cernopendata-client"
        )

    command = ["cernopendata-client", *args]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def fetch_file_locations(
    record_id: str,
    protocol: str = DEFAULT_PROTOCOL,
) -> list[dict[str, Any]]:
    """Fetch file locations for one CERN Open Data record."""
    output = run_cern_client_command(
        [
            "get-file-locations",
            "--recid",
            str(record_id),
            "--protocol",
            protocol,
        ]
    )

    if not output:
        return []

    return [_parse_file_location_line(line) for line in output.splitlines()]


def _parse_file_location_line(line: str) -> dict[str, Any]:
    """Parse one file location output line."""
    file_url = line.strip().split()[0]
    parsed_url = urlparse(file_url)
    file_name = Path(parsed_url.path).name

    return {
        "file_name": file_name,
        "file_url": file_url,
        "file_format": _infer_file_format(file_name),
        "file_size": None,
    }


def enrich_candidate_file_locations(candidate: dict[str, Any]) -> dict[str, Any]:
    """Enrich a candidate with actual file locations from cernopendata-client."""
    record_id = candidate.get("record_id")

    if not record_id:
        candidate["file_location_error"] = "Missing record_id."
        candidate["candidate_score"] = score_candidate(candidate)
        return candidate

    try:
        files = fetch_file_locations(record_id=str(record_id))
        file_formats = sorted(
            {
                file_info.get("file_format", "unknown")
                for file_info in files
            }
        )

        candidate["file_count"] = len(files)
        candidate["file_formats"] = file_formats
        candidate["has_preferred_format"] = bool(set(file_formats) & PREFERRED_FILE_FORMATS)
        candidate["processable_format_count"] = len(set(file_formats) & PROCESSABLE_FILE_FORMATS)
        candidate["files_preview"] = files[:5]
        candidate["file_location_error"] = None

    except Exception as error:
        candidate["file_location_error"] = str(error)

    candidate["candidate_score"] = score_candidate(candidate)

    return candidate


def _fetch_records_api_response(query: str, size: int) -> dict[str, Any]:
    """Call CERN Open Data records API."""
    params = urlencode(
        {
            "q": query,
            "size": size,
        }
    )
    url = f"{CERN_RECORDS_API_URL}?{params}"

    with urlopen(url, timeout=30) as response:
        payload = response.read().decode("utf-8")

    return json.loads(payload)


def _extract_raw_records(raw_response: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract raw record entries from the API response."""
    hits = raw_response.get("hits", {})

    if isinstance(hits, dict):
        raw_records = hits.get("hits", [])
        if isinstance(raw_records, list):
            return raw_records

    return []


def normalize_record_candidate(raw_record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one raw CERN API record into a compact candidate summary."""
    metadata = raw_record.get("metadata", {})

    record_id = str(
        raw_record.get("id")
        or metadata.get("recid")
        or metadata.get("record_id")
        or ""
    )

    files = _extract_files(metadata=metadata)
    file_formats = sorted(
        {
            _infer_file_format(file_info.get("file_name", ""))
            for file_info in files
        }
    )

    candidate = {
        "record_id": record_id,
        "title": metadata.get("title"),
        "description": metadata.get("description"),
        "experiment": _extract_experiment(metadata=metadata),
        "doi": metadata.get("doi"),
        "publication_date": metadata.get("publication_date"),
        "keywords": metadata.get("keywords", []),
        "source_url": f"https://opendata.cern.ch/record/{record_id}" if record_id else None,
        "file_count": len(files),
        "file_formats": file_formats,
        "has_preferred_format": bool(set(file_formats) & PREFERRED_FILE_FORMATS),
        "processable_format_count": len(set(file_formats) & PROCESSABLE_FILE_FORMATS),
        "candidate_score": 0,
        "files_preview": files[:5],
    }

    candidate["candidate_score"] = score_candidate(candidate=candidate)

    return candidate


def _extract_files(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract file metadata when present in the records API response."""
    raw_files = metadata.get("files") or metadata.get("_files") or []

    if not isinstance(raw_files, list):
        return []

    files: list[dict[str, Any]] = []

    for file_entry in raw_files:
        if not isinstance(file_entry, dict):
            continue

        file_name = (
            file_entry.get("key")
            or file_entry.get("filename")
            or file_entry.get("name")
            or file_entry.get("file_name")
        )

        file_size = (
            file_entry.get("size")
            or file_entry.get("filesize")
            or file_entry.get("file_size")
        )

        files.append(
            {
                "file_name": file_name,
                "file_format": _infer_file_format(file_name or ""),
                "file_size": file_size,
            }
        )

    return files


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


def _infer_file_format(file_name: str) -> str:
    """Infer a lightweight file format label from a file name."""
    suffix = Path(file_name).suffix.lower().replace(".", "")

    if not suffix:
        return "unknown"

    return suffix


def score_candidate(candidate: dict[str, Any]) -> int:
    """Score a candidate record for first-pass local ingestion suitability."""
    score = 0

    file_formats = set(candidate.get("file_formats", []))

    if file_formats & {"csv", "json", "parquet"}:
        score += 5

    if "root" in file_formats:
        score += 3

    if candidate.get("file_count", 0) > 0:
        score += 2

    if candidate.get("experiment"):
        score += 1

    if candidate.get("doi"):
        score += 1

    return score


def print_candidate_summary(candidates: list[dict[str, Any]]) -> None:
    """Print candidate records for manual review."""
    if not candidates:
        print("No candidate records found.")
        return

    for index, candidate in enumerate(candidates, start=1):
        print("-" * 80)
        print(f"Candidate #{index}")
        print(f"Record ID: {candidate.get('record_id')}")
        print(f"Title: {candidate.get('title')}")
        print(f"Experiment: {candidate.get('experiment')}")
        print(f"File count: {candidate.get('file_count')}")
        print(f"File formats: {candidate.get('file_formats')}")
        print(f"Candidate score: {candidate.get('candidate_score')}")
        print(f"Source URL: {candidate.get('source_url')}")

        if candidate.get("file_location_error"):
            print(f"File location error: {candidate.get('file_location_error')}")

        files_preview = candidate.get("files_preview", [])
        if files_preview:
            print("Files preview:")
            for file_info in files_preview:
                print(
                    f"  - {file_info.get('file_name')} "
                    f"({file_info.get('file_format')})"
                )


def main() -> None:
    """Run a small CERN Open Data search smoke test."""
    candidates = search_cern_records(
        query=DEFAULT_QUERY,
        size=DEFAULT_RESULT_SIZE,
    )

    print_candidate_summary(candidates=candidates)


if __name__ == "__main__":
    main()