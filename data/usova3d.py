"""USOVA3D dataset loading: a volume-keyed split.json with an annotator/variant mask selector."""
from __future__ import annotations

import json
from pathlib import Path

from .common import DatasetSplits

ANNOTATORS = ("follicle_r1", "follicle_r2", "ovary_r1", "ovary_r2")
VARIANTS = ("binary", "color", "instance")


def _split_images_masks(
    volumes: dict, annotator: str, variant: str, root: Path
) -> tuple[list[str], list[str]]:
    """Match each volume's images to masks by filename stem, dropping unmatched images.

    (The stem-match is done unconditionally here, unlike the notebooks where only the
    ratio-subset helper did stem matching and the full-split loader assumed images/masks
    were already positionally aligned -- stem matching is strictly safer.)
    """
    images, masks = [], []
    for _volume_id, entry in volumes.items():
        image_rels = entry["images"]
        mask_rels = entry.get("labels", {}).get(annotator, {}).get(variant, [])
        mask_lookup = {Path(mr).stem: str(root / mr) for mr in mask_rels}

        for img_rel in image_rels:
            stem = Path(img_rel).stem
            mask_path = mask_lookup.get(stem)
            if mask_path is None:
                continue
            images.append(str(root / img_rel))
            masks.append(mask_path)
    return images, masks


def load_usova3d(
    data_root: str | Path,
    annotator: str = "ovary_r2",
    variant: str = "binary",
    split_file: str = "split.json",
) -> DatasetSplits:
    if annotator not in ANNOTATORS:
        raise ValueError(f"Unknown annotator '{annotator}'. Choices: {ANNOTATORS}")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choices: {VARIANTS}")

    root = Path(data_root)
    with open(root / split_file, encoding="utf-8") as f:
        split = json.load(f)

    volumes_by_split = split["split"]
    image_train, mask_train = _split_images_masks(volumes_by_split["train"], annotator, variant, root)
    image_valid, mask_valid = _split_images_masks(volumes_by_split["val"], annotator, variant, root)
    image_test, mask_test = _split_images_masks(volumes_by_split["test"], annotator, variant, root)

    return DatasetSplits(
        image_train=image_train,
        mask_train=mask_train,
        image_valid=image_valid,
        mask_valid=mask_valid,
        image_test=image_test,
        mask_test=mask_test,
    )
