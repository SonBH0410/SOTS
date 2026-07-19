"""ISIC2018 dataset loading: glob image/mask folders under a dataset root."""
from __future__ import annotations

from pathlib import Path

from .common import DatasetSplits


def load_isic2018(data_root: str | Path) -> DatasetSplits:
    root = Path(data_root)

    image_train = sorted(str(p) for p in root.glob("ISIC2018_Train-images/*.jpg"))
    image_valid = sorted(str(p) for p in root.glob("ISIC2018_Valid-images/*.jpg"))
    image_test = sorted(str(p) for p in root.glob("ISIC2018_Test-images/*.jpg"))

    mask_train = sorted(str(p) for p in root.glob("ISIC2018_Train-mask/*.png"))
    mask_valid = sorted(str(p) for p in root.glob("ISIC2018_Valid-mask/*.png"))
    mask_test = sorted(str(p) for p in root.glob("ISIC2018_Test-mask/*.png"))

    return DatasetSplits(
        image_train=image_train,
        mask_train=mask_train,
        image_valid=image_valid,
        mask_valid=mask_valid,
        image_test=image_test,
        mask_test=mask_test,
    )
