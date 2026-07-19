"""OTU2D dataset loading: a flat JSON annotation file with a 'split' key per record."""
from __future__ import annotations

import json
from pathlib import Path

from .common import DatasetSplits


def load_otu2d(data_root: str | Path, annotation_file: str = "OTU_2D/OTU_2D_annotation.json") -> DatasetSplits:
    root = Path(data_root)
    with open(root / annotation_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    image_train, mask_train = [], []
    image_valid, mask_valid = [], []
    image_test, mask_test = [], []

    for item in records:
        img_path = str(root / str(item["file_path_img"]))
        mask_path = str(root / str(item["file_path_ann"]))
        split = item["split"]
        if split == "train":
            image_train.append(img_path)
            mask_train.append(mask_path)
        elif split == "validation":
            image_valid.append(img_path)
            mask_valid.append(mask_path)
        elif split == "test":
            image_test.append(img_path)
            mask_test.append(mask_path)

    return DatasetSplits(
        image_train=image_train,
        mask_train=mask_train,
        image_valid=image_valid,
        mask_valid=mask_valid,
        image_test=image_test,
        mask_test=mask_test,
    )
