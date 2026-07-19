"""Projection/predictor heads.

`mlp_head` is a single generic builder that covers every method's MLP head shape found in the
notebooks (BYOL projector/predictor, SimCLR projector, SimSiam projector/predictor, MoCo's fused
projection): N hidden `Dense(no_bias) -> BatchNorm -> ReLU` layers, then a final `Dense`, optionally
followed by a trailing BatchNorm (with or without a learnable scale).

`projection_head`/`fuse_with_skips`/`fuse_with_skips_vector` are Barlow Twins' own functional-style
pieces (operate on a KerasTensor directly, matching how `models/encoders.py` wires up
`build_barlow_encoder`), kept separate since that encoder's richer architecture is specific to that
method.
"""
from __future__ import annotations

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.regularizers import l2


def mlp_head(
    input_dim: int,
    hidden_dims: list[int],
    out_dim: int,
    final_bn: bool = False,
    final_bn_scale: bool = True,
    name: str = "head",
) -> tf.keras.Sequential:
    layers = []
    for i, hidden_dim in enumerate(hidden_dims, start=1):
        layers.append(L.Dense(hidden_dim, use_bias=False, name=f"{name}_fc{i}"))
        layers.append(L.BatchNormalization(name=f"{name}_bn{i}"))
        layers.append(L.ReLU(name=f"{name}_relu{i}"))

    final_idx = len(hidden_dims) + 1
    layers.append(L.Dense(out_dim, use_bias=not final_bn, name=f"{name}_fc{final_idx}"))
    if final_bn:
        layers.append(L.BatchNormalization(scale=final_bn_scale, name=f"{name}_bn{final_idx}"))

    model = tf.keras.Sequential(layers, name=name)
    model.build((None, input_dim))
    return model


def projection_head(x, hidden_dim: int = 128, weight_decay: float = 1e-6):
    """Barlow Twins' functional-graph projection head (2 hidden layers + output Dense)."""
    for i in range(2):
        x = L.Dense(hidden_dim, name=f"projection_layer_{i}", kernel_regularizer=l2(weight_decay))(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("relu")(x)
    return L.Dense(hidden_dim, name="projection_output")(x)


def fuse_with_skips(x, skip_list, out_channels: int = 512):
    """Resize+concat a set of skip-connection feature maps onto x, then project back down."""
    target_shape = x.shape[1:3]
    skips_resized = [L.Resizing(*target_shape)(s) for s in skip_list]
    fused = L.Concatenate(axis=-1)([x] + skips_resized)
    fused = L.Conv2D(out_channels, 1, padding="same")(fused)
    fused = L.BatchNormalization()(fused)
    fused = L.ReLU()(fused)
    return fused


def fuse_with_skips_vector(x, skip_list):
    """Pooled-vector alternative to fuse_with_skips (GAP each skip map, then concat)."""
    v_main = L.GlobalAveragePooling2D()(x)
    v_skips = [L.GlobalAveragePooling2D()(s) for s in skip_list]
    return L.Concatenate(axis=-1)([v_main] + v_skips)
