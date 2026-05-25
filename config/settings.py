from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    storage_mode: str
    project_root: Path
    data_dir: Path
    models_dir: Path
    evaluation_metrics_dir: Path
    evaluation_reports_dir: Path
    s3_bucket: str | None
    s3_prefix: str
    dataset_mode: str

def get_settings()->Settings:

    storage_mode = os.getenv("COLLIDEROPS_STORAGE_MODE", "local").lower()

    return Settings(
        storage_mode=storage_mode,
        project_root=PROJECT_ROOT,
        data_dir=PROJECT_ROOT / "data",
        models_dir=PROJECT_ROOT / "models",
        evaluation_metrics_dir= PROJECT_ROOT / "evaluation_metrics",
        evaluation_reports_dir= PROJECT_ROOT / "evaluation_reports",
        s3_bucket= os.getenv("COLLIDEROPS_S3_BUCKET"),
        s3_prefix=os.getenv("COLLIDEROPS_S3_PREFIX", "dev"),
        dataset_mode="curated_higgs"
    )

