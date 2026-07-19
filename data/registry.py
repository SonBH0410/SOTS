"""Dataset name -> loader dispatch."""
from __future__ import annotations

from .common import DatasetSplits
from .isic2018 import load_isic2018
from .otu2d import load_otu2d
from .usova3d import load_usova3d

DATASETS = ("isic2018", "otu2d", "usova3d")


def load_dataset(
    name: str,
    data_root: str,
    annotator: str = "ovary_r2",
    variant: str = "binary",
) -> DatasetSplits:
    if name == "isic2018":
        return load_isic2018(data_root)
    if name == "otu2d":
        return load_otu2d(data_root)
    if name == "usova3d":
        return load_usova3d(data_root, annotator=annotator, variant=variant)
    raise ValueError(f"Unknown dataset '{name}'. Choices: {DATASETS}")
