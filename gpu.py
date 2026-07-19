"""GPU visibility and memory-growth setup, ported from the notebooks' GPU-setup cell."""
from __future__ import annotations

import random

import numpy as np
import tensorflow as tf


def setup_gpu(gpu_index: int = 0) -> None:
    """Enable memory growth on every physical GPU, then pin visibility to one of them.

    Memory growth must be set on ALL physical GPUs before restricting visibility,
    otherwise TensorFlow allocates the full memory pool on the others.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU found, running on CPU.")
        return

    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

    index = min(gpu_index, len(gpus) - 1)
    tf.config.set_visible_devices(gpus[index], "GPU")
    print(f"Using GPU {index} of {len(gpus)} with memory growth enabled")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
