"""ROOT file inspection adapter for ColliderOpsAI.

This adapter is the first step toward real CERN/Open Data ingestion.

Current scope:
- Read the local dataset registry.
- Select a ROOT file URL from a registered CERN Open Data record.
- Open the ROOT file with uproot.
- Inspect available keys/classes.
- Identify tree-like objects when available.
- Print a compact file structure summary.

This module intentionally does not perform full ETL yet. The first goal is to
verify whether selected CERN ROOT files can be inspected with uproot directly or
whether they require heavier CMS/CMSSW tooling.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_REGISTRY_PATH = PROJECT_ROOT / "data" / "dataset_registry.json"
DEFAULT_RECORD_ID = "7901"
DEFAULT_FILE_INDEX = 0
DEFAULT_TREE_NAME = "Events;1"

DEFAULT_BRANCH_PATTERNS = [
    "EventAuxiliary",
    "GenEventInfoProduct",
    "recoGenJets",
    "genParticles",
]

# Rich feature candidate patterns for branch classification
RICH_FEATURE_CANDIDATE_PATTERNS = {
    "photon": ["Photon", "photon", "gedPhoton", "recoPhoton"],
    "electron": ["Electron", "electron", "GsfElectron", "gedGsfElectron"],
    "muon": ["Muon", "muon"],
    "jet": ["Jet", "jet", "GenJet", "PFJet", "ak5", "ak7"],
    "met": ["MET", "met", "MissingET", "CaloMET", "PFMET"],
    "gen_particle": ["GenParticle", "genParticle", "GenPart", "genParticles"],
    "kinematics": ["pt", "eta", "phi", "mass", "energy", "momentum"],
    "trigger": ["Trigger", "trigger", "HLT", "L1"],
    "lhe": ["LHE", "lhe"],
    "event_info": ["Event", "Run", "Luminosity", "Pileup", "Vertex", "rho"],
}

READABLE_EVENT_BRANCHES = {
    "gen_event_present": "GenEventInfoProduct_generator__SIM./GenEventInfoProduct_generator__SIM.present",
    "gen_event_obj": "GenEventInfoProduct_generator__SIM./GenEventInfoProduct_generator__SIM.obj",
    "gen_particles_present": "ints_genParticles__SIM./ints_genParticles__SIM.present",
    "gen_particles_obj": "ints_genParticles__SIM./ints_genParticles__SIM.obj",
    "ak5_genjets_present": "recoGenJets_ak5GenJets__SIM./recoGenJets_ak5GenJets__SIM.present",
}

DEFAULT_SAMPLE_ROWS = 5


try:
    import uproot
except ImportError:  # pragma: no cover - handled at runtime
    uproot = None


def ensure_uproot_available() -> None:
    """Raise an error if uproot is not installed."""
    if uproot is None:
        raise RuntimeError(
            "uproot is not installed. Install it with: pip install uproot awkward"
        )


def load_dataset_registry() -> list[dict[str, Any]]:
    """Load registered CERN/Open Data dataset metadata."""
    if not DATASET_REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Dataset registry not found: {DATASET_REGISTRY_PATH}. "
            "Run etl.cern_client.register_cern_record first."
        )

    with DATASET_REGISTRY_PATH.open("r", encoding="utf-8") as file:
        registry = json.load(file)

    if isinstance(registry, list):
        return registry

    return [registry]


def get_registered_record(record_id: str = DEFAULT_RECORD_ID) -> dict[str, Any]:
    """Return one registered dataset record from data/dataset_registry.json."""
    registry = load_dataset_registry()

    for record in registry:
        if str(record.get("cern_record_id")) == str(record_id):
            return record

    raise ValueError(f"Record {record_id} not found in dataset registry.")


def get_root_file_url(
    record_id: str = DEFAULT_RECORD_ID,
    file_index: int = DEFAULT_FILE_INDEX,
) -> str:
    """Return a ROOT file URL from a registered CERN Open Data record."""
    record = get_registered_record(record_id=record_id)
    files = record.get("files", [])

    root_files = [
        file_info for file_info in files
        if file_info.get("file_format") == "root"
    ]

    if not root_files:
        raise ValueError(f"No ROOT files found for registered record {record_id}.")

    if file_index >= len(root_files):
        raise IndexError(
            f"Requested file_index={file_index}, but only {len(root_files)} ROOT files are available."
        )

    file_url = root_files[file_index].get("file_url")

    if not file_url:
        raise ValueError(
            f"ROOT file at index {file_index} for record {record_id} has no file_url."
        )

    return file_url


def open_root_file(file_path_or_url: str):
    """Open a ROOT file from a local path or URL using uproot."""
    ensure_uproot_available()
    return uproot.open(file_path_or_url)


def list_root_keys(file_path_or_url: str) -> list[str]:
    """List top-level keys in a ROOT file."""
    with open_root_file(file_path_or_url) as root_file:
        return list(root_file.keys())


def inspect_root_file(file_path_or_url: str, max_keys: int = 25) -> dict[str, Any]:
    """Inspect a ROOT file and return a compact structure summary."""
    ensure_uproot_available()

    summary: dict[str, Any] = {
        "file_path_or_url": file_path_or_url,
        "top_level_key_count": 0,
        "keys": [],
        "tree_candidates": [],
        "status": "success",
        "error": None,
    }

    try:
        with uproot.open(file_path_or_url) as root_file:
            keys = list(root_file.keys())
            summary["top_level_key_count"] = len(keys)
            summary["keys"] = keys[:max_keys]

            tree_candidates = []

            for key in keys[:max_keys]:
                try:
                    obj = root_file[key]
                    class_name = obj.classname

                    candidate = {
                        "key": key,
                        "class_name": class_name,
                    }

                    if hasattr(obj, "num_entries"):
                        candidate["num_entries"] = obj.num_entries
                        candidate["branches_preview"] = list(obj.keys())[:25]
                        tree_candidates.append(candidate)

                except Exception as object_error:
                    tree_candidates.append(
                        {
                            "key": key,
                            "class_name": "unknown",
                            "error": str(object_error),
                        }
                    )

            summary["tree_candidates"] = tree_candidates

    except Exception as error:
        summary["status"] = "error"
        summary["error"] = str(error)

    return summary


def inspect_registered_root_file(
    record_id: str = DEFAULT_RECORD_ID,
    file_index: int = DEFAULT_FILE_INDEX,
    max_keys: int = 25,
) -> dict[str, Any]:
    """Inspect one ROOT file from a registered CERN/Open Data record."""
    file_url = get_root_file_url(
        record_id=record_id,
        file_index=file_index,
    )

    return inspect_root_file(
        file_path_or_url=file_url,
        max_keys=max_keys,
    )


def get_tree(
    file_path_or_url: str,
    tree_name: str = DEFAULT_TREE_NAME,
):
    """Return a tree-like object from a ROOT file."""
    ensure_uproot_available()
    root_file = uproot.open(file_path_or_url)

    if tree_name not in root_file:
        available_keys = list(root_file.keys())
        root_file.close()
        raise KeyError(
            f"Tree {tree_name} not found. Available keys: {available_keys}"
        )

    return root_file, root_file[tree_name]


def list_tree_branches(
    file_path_or_url: str,
    tree_name: str = DEFAULT_TREE_NAME,
) -> list[str]:
    """List branches from a selected ROOT tree."""
    root_file, tree = get_tree(
        file_path_or_url=file_path_or_url,
        tree_name=tree_name,
    )

    try:
        return list(tree.keys())
    finally:
        root_file.close()


def find_branches_by_pattern(
    branches: list[str],
    patterns: list[str] | None = None,
) -> list[str]:
    """Return branches containing any selected pattern."""
    selected_patterns = patterns or DEFAULT_BRANCH_PATTERNS

    return [
        branch for branch in branches
        if any(pattern in branch for pattern in selected_patterns)
    ]


def sample_event_branch(
    file_path_or_url: str,
    branch_name: str,
    tree_name: str = DEFAULT_TREE_NAME,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> dict[str, Any]:
    """Attempt to read a tiny sample from one branch."""
    root_file, tree = get_tree(
        file_path_or_url=file_path_or_url,
        tree_name=tree_name,
    )

    try:
        array = tree[branch_name].array(
            entry_start=0,
            entry_stop=sample_rows,
            library="ak",
        )

        return {
            "branch_name": branch_name,
            "status": "success",
            "sample_type": type(array).__name__,
            "sample_preview": str(array[:sample_rows])[:500],
            "error": None,
        }

    except Exception as error:
        return {
            "branch_name": branch_name,
            "status": "error",
            "sample_type": None,
            "sample_preview": None,
            "error": str(error),
        }

    finally:
        root_file.close()


def probe_event_branch_extraction(
    file_path_or_url: str,
    tree_name: str = DEFAULT_TREE_NAME,
    branch_patterns: list[str] | None = None,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    max_branches: int = 10,
) -> dict[str, Any]:
    """Probe which event branches can be read with uproot."""
    summary: dict[str, Any] = {
        "file_path_or_url": file_path_or_url,
        "tree_name": tree_name,
        "branch_patterns": branch_patterns or DEFAULT_BRANCH_PATTERNS,
        "total_branch_count": 0,
        "matched_branch_count": 0,
        "sampled_branch_count": 0,
        "successful_branch_count": 0,
        "failed_branch_count": 0,
        "matched_branches_preview": [],
        "branch_samples": [],
        "status": "success",
        "error": None,
    }

    try:
        branches = list_tree_branches(
            file_path_or_url=file_path_or_url,
            tree_name=tree_name,
        )
        matched_branches = find_branches_by_pattern(
            branches=branches,
            patterns=branch_patterns,
        )
        branches_to_sample = matched_branches[:max_branches]

        branch_samples = [
            sample_event_branch(
                file_path_or_url=file_path_or_url,
                branch_name=branch_name,
                tree_name=tree_name,
                sample_rows=sample_rows,
            )
            for branch_name in branches_to_sample
        ]

        successful_branch_count = sum(
            1 for sample in branch_samples if sample.get("status") == "success"
        )
        failed_branch_count = sum(
            1 for sample in branch_samples if sample.get("status") == "error"
        )

        summary.update(
            {
                "total_branch_count": len(branches),
                "matched_branch_count": len(matched_branches),
                "sampled_branch_count": len(branch_samples),
                "successful_branch_count": successful_branch_count,
                "failed_branch_count": failed_branch_count,
                "matched_branches_preview": matched_branches[:25],
                "branch_samples": branch_samples,
            }
        )

    except Exception as error:
        summary["status"] = "error"
        summary["error"] = str(error)

    return summary


def probe_registered_event_branch_extraction(
    record_id: str = DEFAULT_RECORD_ID,
    file_index: int = DEFAULT_FILE_INDEX,
    tree_name: str = DEFAULT_TREE_NAME,
    branch_patterns: list[str] | None = None,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    max_branches: int = 10,
) -> dict[str, Any]:
    """Probe branch-level extraction for one registered CERN ROOT file."""
    file_url = get_root_file_url(
        record_id=record_id,
        file_index=file_index,
    )

    return probe_event_branch_extraction(
        file_path_or_url=file_url,
        tree_name=tree_name,
        branch_patterns=branch_patterns,
        sample_rows=sample_rows,
        max_branches=max_branches,
    )



# ---- Rich feature branch candidate utilities ----

def classify_branch_candidate(
    branch_name: str,
    candidate_patterns: dict[str, list[str]] | None = None,
) -> list[str]:
    """Classify a branch into one or more candidate feature groups."""
    selected_patterns = candidate_patterns or RICH_FEATURE_CANDIDATE_PATTERNS
    matched_groups: list[str] = []

    for group_name, patterns in selected_patterns.items():
        if any(pattern in branch_name for pattern in patterns):
            matched_groups.append(group_name)

    return matched_groups


def discover_rich_feature_branch_candidates(
    file_path_or_url: str,
    record_id: str = DEFAULT_RECORD_ID,
    label: str | None = None,
    file_index: int | None = None,
    tree_name: str = DEFAULT_TREE_NAME,
    candidate_patterns: dict[str, list[str]] | None = None,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    max_candidates: int | None = 250,
) -> list[dict[str, Any]]:
    """Discover richer readable branch candidates from a ROOT tree.

    This does not add branches to the training feature set. It creates a report of
    candidate branches, grouped by physics-style keywords, and tests whether a
    tiny sample can be read with uproot.
    """
    branches = list_tree_branches(
        file_path_or_url=file_path_or_url,
        tree_name=tree_name,
    )

    candidate_rows: list[dict[str, Any]] = []

    for branch_name in branches:
        matched_groups = classify_branch_candidate(
            branch_name=branch_name,
            candidate_patterns=candidate_patterns,
        )

        if not matched_groups:
            continue

        sample = sample_event_branch(
            file_path_or_url=file_path_or_url,
            branch_name=branch_name,
            tree_name=tree_name,
            sample_rows=sample_rows,
        )

        candidate_rows.append(
            {
                "record_id": str(record_id),
                "label": label,
                "file_index": file_index,
                "file_path_or_url": file_path_or_url,
                "tree_name": tree_name,
                "branch_name": branch_name,
                "candidate_groups": ",".join(matched_groups),
                "status": sample.get("status"),
                "sample_type": sample.get("sample_type"),
                "sample_preview": sample.get("sample_preview"),
                "error": sample.get("error"),
            }
        )

        if max_candidates is not None and len(candidate_rows) >= max_candidates:
            break

    return candidate_rows


def discover_registered_rich_feature_branch_candidates(
    record_id: str = DEFAULT_RECORD_ID,
    label: str | None = None,
    file_index: int = DEFAULT_FILE_INDEX,
    tree_name: str = DEFAULT_TREE_NAME,
    candidate_patterns: dict[str, list[str]] | None = None,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    max_candidates: int | None = 250,
) -> list[dict[str, Any]]:
    """Discover richer branch candidates from a registered CERN ROOT file URL."""
    file_url = get_root_file_url(
        record_id=record_id,
        file_index=file_index,
    )

    return discover_rich_feature_branch_candidates(
        file_path_or_url=file_url,
        record_id=record_id,
        label=label,
        file_index=file_index,
        tree_name=tree_name,
        candidate_patterns=candidate_patterns,
        sample_rows=sample_rows,
        max_candidates=max_candidates,
    )


def print_rich_feature_candidate_summary(
    candidate_rows: list[dict[str, Any]],
    max_rows: int = 25,
) -> None:
    """Print a compact summary of rich feature branch candidates."""
    print("-" * 80)
    print(f"Rich feature branch candidates: {len(candidate_rows)}")

    if not candidate_rows:
        print("No rich feature candidates found.")
        return

    success_count = sum(1 for row in candidate_rows if row.get("status") == "success")
    failed_count = sum(1 for row in candidate_rows if row.get("status") == "error")
    print(f"Readable candidates: {success_count}")
    print(f"Failed candidates: {failed_count}")

    print("Candidates preview:")
    for row in candidate_rows[:max_rows]:
        print(f"  - Branch: {row.get('branch_name')}")
        print(f"    Groups: {row.get('candidate_groups')}")
        print(f"    Status: {row.get('status')}")
        if row.get("sample_type"):
            print(f"    Sample type: {row.get('sample_type')}")
        if row.get("sample_preview"):
            print(f"    Sample preview: {row.get('sample_preview')}")
        if row.get("error"):
            print(f"    Error: {row.get('error')}")


def print_branch_probe_summary(summary: dict[str, Any]) -> None:
    """Print branch-level extraction probe results."""
    print("-" * 80)
    print(f"Branch probe file: {summary.get('file_path_or_url')}")
    print(f"Tree: {summary.get('tree_name')}")
    print(f"Status: {summary.get('status')}")

    if summary.get("error"):
        print(f"Error: {summary.get('error')}")
        return

    print(f"Total branches: {summary.get('total_branch_count')}")
    print(f"Matched branches: {summary.get('matched_branch_count')}")
    print(f"Sampled branches: {summary.get('sampled_branch_count')}")
    print(f"Successful samples: {summary.get('successful_branch_count')}")
    print(f"Failed samples: {summary.get('failed_branch_count')}")

    matched_branches_preview = summary.get("matched_branches_preview", [])
    if matched_branches_preview:
        print("Matched branches preview:")
        for branch in matched_branches_preview:
            print(f"  - {branch}")

    branch_samples = summary.get("branch_samples", [])
    if branch_samples:
        print("Branch samples:")
        for sample in branch_samples:
            print(f"  - Branch: {sample.get('branch_name')}")
            print(f"    Status: {sample.get('status')}")
            if sample.get("sample_type"):
                print(f"    Sample type: {sample.get('sample_type')}")
            if sample.get("sample_preview"):
                print(f"    Sample preview: {sample.get('sample_preview')}")
            if sample.get("error"):
                print(f"    Error: {sample.get('error')}")


# ---- Event feature extraction utilities ----

def _to_python_scalar(value: Any) -> Any:
    """Convert awkward/numpy scalar-like values into Python-native values when possible."""
    if value is None:
        return None

    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass

    if isinstance(value, float) and math.isnan(value):
        return None

    return value


def _safe_len(value: Any) -> int | None:
    """Return len(value) when possible, otherwise None."""
    try:
        return len(value)
    except Exception:
        return None


def _to_float_list(value: Any) -> list[float]:
    """Convert a list-like awkward/numpy/Python value into a list of floats."""
    if value is None:
        return []

    python_value = _to_python_scalar(value)

    if python_value is None:
        return []

    if not isinstance(python_value, list):
        python_value = [python_value]

    float_values: list[float] = []

    for item in python_value:
        try:
            converted_item = float(item)
            if not math.isnan(converted_item):
                float_values.append(converted_item)
        except Exception:
            continue

    return float_values


def _numeric_summary(values: list[float], prefix: str) -> dict[str, Any]:
    """Return basic non-leaky summary statistics for a numeric list."""
    if not values:
        return {
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_sum": None,
            f"{prefix}_unique_count": 0,
        }

    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count

    return {
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
        f"{prefix}_mean": mean,
        f"{prefix}_std": math.sqrt(variance),
        f"{prefix}_sum": sum(values),
        f"{prefix}_unique_count": len(set(values)),
    }


def _extract_optional_field(record: Any, field_name: str) -> Any:
    """Extract a field from an awkward record when available."""
    try:
        if hasattr(record, "fields") and field_name in record.fields:
            return _to_python_scalar(record[field_name])
    except Exception:
        return None

    return None


def extract_readable_event_features(
    file_path_or_url: str,
    record_id: str = DEFAULT_RECORD_ID,
    tree_name: str = DEFAULT_TREE_NAME,
    label: str | None = None,
    max_events: int = 100,
) -> list[dict[str, Any]]:
    """Extract a small event-level feature table from readable ROOT branches.

    This function intentionally extracts only branches already proven readable by the
    branch probe. It does not attempt to deserialize complex CMS EDM objects such as
    reco::GenJet collections.
    `gen_event_signal_process_id` is retained for traceability but excluded from curated model features because it is leakage-prone.
    """
    ensure_uproot_available()

    root_file, tree = get_tree(
        file_path_or_url=file_path_or_url,
        tree_name=tree_name,
    )

    try:
        available_branches = set(tree.keys())
        selected_branches = {
            alias: branch_name
            for alias, branch_name in READABLE_EVENT_BRANCHES.items()
            if branch_name in available_branches
        }

        if not selected_branches:
            raise ValueError("None of the configured readable event branches are available.")

        arrays = tree.arrays(
            list(selected_branches.values()),
            entry_start=0,
            entry_stop=max_events,
            library="ak",
        )

        event_count = len(arrays)
        rows: list[dict[str, Any]] = []

        for event_index in range(event_count):
            row: dict[str, Any] = {
                "event_index": event_index,
                "source_record_id": str(record_id),
                "source_file_url": file_path_or_url,
                "label": label,
            }

            for alias, branch_name in selected_branches.items():
                value = arrays[branch_name][event_index]

                if alias.endswith("_present"):
                    row[alias] = bool(_to_python_scalar(value))
                elif alias == "gen_particles_obj":
                    particle_values = _to_float_list(value)
                    row["gen_particle_count"] = len(particle_values)
                    row.update(
                        _numeric_summary(
                            values=particle_values,
                            prefix="gen_particle_id",
                        )
                    )
                elif alias == "gen_event_obj":
                    weights = _to_float_list(
                        _extract_optional_field(value, "weights_")
                    )
                    row["gen_event_weight_count"] = len(weights)
                    row.update(
                        _numeric_summary(
                            values=weights,
                            prefix="gen_event_weight",
                        )
                    )
                    row["gen_event_signal_process_id"] = _extract_optional_field(
                        value, "signalProcessID_"
                    )
                    row["gen_event_qscale"] = _extract_optional_field(value, "qScale_")
                else:
                    row[alias] = str(value)[:500]

            rows.append(row)

        return rows

    finally:
        root_file.close()


def extract_registered_root_event_features(
    record_id: str = DEFAULT_RECORD_ID,
    file_index: int = DEFAULT_FILE_INDEX,
    tree_name: str = DEFAULT_TREE_NAME,
    label: str | None = "signal",
    max_events: int = 100,
) -> list[dict[str, Any]]:
    """Extract readable event features from one registered CERN ROOT file."""
    file_url = get_root_file_url(
        record_id=record_id,
        file_index=file_index,
    )

    return extract_readable_event_features(
        file_path_or_url=file_url,
        record_id=record_id,
        tree_name=tree_name,
        label=label,
        max_events=max_events,
    )


def print_extracted_feature_summary(rows: list[dict[str, Any]], max_rows: int = 5) -> None:
    """Print a compact summary of extracted event features."""
    print("-" * 80)
    print(f"Extracted event feature rows: {len(rows)}")

    if not rows:
        print("No rows extracted.")
        return

    print(f"Columns: {list(rows[0].keys())}")
    print("Rows preview:")
    for row in rows[:max_rows]:
        print(row)


def print_root_inspection_summary(summary: dict[str, Any]) -> None:
    """Print ROOT file inspection results."""
    print("-" * 80)
    print(f"ROOT file: {summary.get('file_path_or_url')}")
    print(f"Status: {summary.get('status')}")

    if summary.get("error"):
        print(f"Error: {summary.get('error')}")
        return

    print(f"Top-level key count: {summary.get('top_level_key_count')}")

    print("Top-level keys preview:")
    for key in summary.get("keys", []):
        print(f"  - {key}")

    tree_candidates = summary.get("tree_candidates", [])

    if not tree_candidates:
        print("No tree-like candidates found in the inspected key preview.")
        return

    print("Tree-like candidates:")
    for candidate in tree_candidates:
        print(f"  - Key: {candidate.get('key')}")
        print(f"    Class: {candidate.get('class_name')}")

        if "num_entries" in candidate:
            print(f"    Entries: {candidate.get('num_entries')}")

        branches_preview = candidate.get("branches_preview", [])
        if branches_preview:
            print("    Branches preview:")
            for branch in branches_preview:
                print(f"      - {branch}")

        if candidate.get("error"):
            print(f"    Error: {candidate.get('error')}")


def main() -> None:
    """Inspect and probe the first ROOT file from the default registered CERN record."""
    inspection_summary = inspect_registered_root_file(
        record_id=DEFAULT_RECORD_ID,
        file_index=DEFAULT_FILE_INDEX,
    )
    print_root_inspection_summary(summary=inspection_summary)

    branch_probe_summary = probe_registered_event_branch_extraction(
        record_id=DEFAULT_RECORD_ID,
        file_index=DEFAULT_FILE_INDEX,
    )
    print_branch_probe_summary(summary=branch_probe_summary)

    extracted_rows = extract_registered_root_event_features(
        record_id=DEFAULT_RECORD_ID,
        file_index=DEFAULT_FILE_INDEX,
        label="signal",
        max_events=10,
    )
    print_extracted_feature_summary(rows=extracted_rows)

    candidate_rows = discover_registered_rich_feature_branch_candidates(
        record_id=DEFAULT_RECORD_ID,
        label="signal",
        file_index=DEFAULT_FILE_INDEX,
        max_candidates=25,
    )
    print_rich_feature_candidate_summary(candidate_rows=candidate_rows)


if __name__ == "__main__":
    main()