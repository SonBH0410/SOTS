"""Shared dataset plumbing: the DatasetSplits container, image/mask loading, array building,
and a single deterministic label-ratio subsampler used by all three datasets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class DatasetSplits:
    image_train: list[str]
    mask_train: list[str]
    image_valid: list[str] = field(default_factory=list)
    mask_valid: list[str] = field(default_factory=list)
    image_test: list[str] = field(default_factory=list)
    mask_test: list[str] = field(default_factory=list)


def load_image_mask(image_path: str, mask_path: str, img_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Read an image/mask pair, resize, normalize to [0,1], binarize the mask."""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (img_size, img_size))
    mask = cv2.resize(mask, (img_size, img_size))

    image = image / 255.0
    mask = mask / 255.0
    mask[mask > 0.0] = 1.0
    mask = np.expand_dims(mask, axis=-1)

    return image.astype(np.float32), mask.astype(np.float32)


def build_arrays(
    image_paths: list[str], mask_paths: list[str], img_size: int = 256, desc: str = "Loading"
) -> tuple[np.ndarray, np.ndarray]:
    """Load a list of image/mask path pairs into stacked float32 numpy arrays."""
    x = np.zeros((len(image_paths), img_size, img_size, 3), dtype=np.float32)
    y = np.zeros((len(mask_paths), img_size, img_size, 1), dtype=np.float32)
    for i, (img_path, mask_path) in enumerate(tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc=desc)):
        img, mask = load_image_mask(img_path, mask_path, img_size)
        x[i] = img
        y[i] = mask
    return x, y


def build_image_array(image_paths: list[str], img_size: int = 256, desc: str = "Loading images") -> np.ndarray:
    """Load images only (no masks) -- used for SSL pretraining, which never touches labels."""
    x = np.zeros((len(image_paths), img_size, img_size, 3), dtype=np.float32)
    for i, img_path in enumerate(tqdm(image_paths, desc=desc)):
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_size, img_size))
        x[i] = (image / 255.0).astype(np.float32)
    return x


def get_label_subset(
    image_paths: list[str], mask_paths: list[str], ratio: float
) -> tuple[list[str], list[str]]:
    """Deterministic, evenly-spaced label-efficiency subset shared by all three datasets.

    Uses np.linspace index sampling (rather than a naive head-slice) so the subset spreads
    across the full training set instead of always being its first N items.
    """
    if ratio >= 1.0:
        return list(image_paths), list(mask_paths)

    n = max(1, round(len(image_paths) * ratio))
    indices = np.linspace(0, len(image_paths) - 1, n, dtype=int)
    return [image_paths[i] for i in indices], [mask_paths[i] for i in indices]
