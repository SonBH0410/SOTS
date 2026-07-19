"""Default dataset roots (as hardcoded in the original notebooks) and output path conventions."""
from __future__ import annotations

import datetime
from pathlib import Path

# Original per-notebook hardcoded roots. Override per-run with --data-root.
DATASET_DEFAULTS = {
    "isic2018": "/home/sonbh/UTBT_Dataset/ISIC2018",
    "otu2d": "/home/sonbh/UTBT_Dataset",
    "usova3d": "/mnt/nvme0/home/sonbh/Dataset/USOVA3D_Dataset",
}


def default_data_root(dataset: str) -> str:
    try:
        return DATASET_DEFAULTS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset '{dataset}'. Choices: {list(DATASET_DEFAULTS)}") from exc


def run_dir(output_dir: Path, dataset: str, method: str) -> Path:
    """outputs/{dataset}/{method}/ -- created if missing."""
    d = Path(output_dir) / dataset / method
    d.mkdir(parents=True, exist_ok=True)
    return d


def timestamped_run_name(run_name: str | None) -> str:
    if run_name:
        return run_name
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
