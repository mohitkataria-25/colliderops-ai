"""Reusable schema validation helpers for ColliderOpsAI ETL jobs."""

from __future__ import annotations


REQUIRED_COLLIDER_COLUMNS = [
    "event_id",
    "label",
    "DER_mass_MMC",
    "DER_mass_transverse_met_lep",
    "DER_mass_vis",
    "PRI_tau_pt",
    "PRI_lep_pt",
]


def validate_required_columns(df, required_columns: list[str]) -> None:
    """Validate that a Spark DataFrame contains all required columns."""
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")


def validate_collider_schema(df) -> None:
    """Validate the expected ColliderOpsAI event schema."""
    validate_required_columns(
        df=df,
        required_columns=REQUIRED_COLLIDER_COLUMNS,
    )