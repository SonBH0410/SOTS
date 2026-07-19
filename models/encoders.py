"""Per-method encoder builders, and the single shared segmentation-model builder.

Every encoder below is built on top of `VGG16_encoder_branch`, so the named VGG16 layers
(block1_conv2 .. block5_conv3) always exist in the resulting Keras graph no matter what
projection/prediction head is appended after them. `build_segmentation_model` exploits that: it
looks the skip-connection layers up **by name**, so it works unmodified for a Barlow Twins encoder,
a MoCo query encoder, a bare ImageNet-only baseline, or anything else built from this module -- unlike
the original notebooks, which sliced each encoder by a hand-counted negative layer index (`-16` for
Barlow Twins, `-7` for MoCo) that silently breaks the moment the head architecture changes.
"""
from __future__ import annotations

import tensorflow as tf
import tensorflow.keras.layers as L

from .heads import fuse_with_skips, mlp_head, projection_head
from .unet_parts import (
    SpatialPyramidPoolingFast,
    VGG16_encoder_branch,
    decoder_branch_normal,
    output_branch,
)

SKIP_LAYER_NAMES = ("block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3")
BOTTLENECK_LAYER_NAME = "block5_conv3"


def build_pooled_encoder(input_shape=(256, 256, 3), name: str = "pooled_encoder") -> tf.keras.Model:
    """VGG16 trunk -> global average pool. Shared base for BYOL, SimCLR, SimSiam, and the baseline."""
    inputs = tf.keras.Input(input_shape)
    x, _ = VGG16_encoder_branch(inputs)
    x = L.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, x, name=name)


def build_barlow_encoder(
    input_shape=(256, 256, 3), hidden_dim: int = 256, weight_decay: float = 1e-6
) -> tf.keras.Model:
    """VGG16 trunk -> SPPF -> skip-fusion -> GAP -> dense projection head (Barlow Twins' own recipe)."""
    inputs = tf.keras.Input(input_shape)
    x, skip_connections = VGG16_encoder_branch(inputs)
    x = SpatialPyramidPoolingFast(x, pool_size=(5, 5, 5))
    fused = fuse_with_skips(x, skip_connections[1:4], out_channels=512)
    x = L.GlobalAveragePooling2D()(fused)
    outputs = projection_head(x, hidden_dim=hidden_dim, weight_decay=weight_decay)
    return tf.keras.Model(inputs, outputs, name="barlow_encoder")


def build_moco_encoder(
    input_shape=(256, 256, 3), proj_dim: int = 256, hidden_dim: int = 512, name: str = "moco_encoder"
) -> tf.keras.Model:
    """VGG16 trunk -> GAP -> fused 2-layer projection (MoCo v2's "backbone", momentum-updated whole)."""
    inputs = tf.keras.Input(input_shape)
    x, _ = VGG16_encoder_branch(inputs)
    x = L.GlobalAveragePooling2D()(x)
    proj = mlp_head(x.shape[-1], [hidden_dim], proj_dim, final_bn=True, final_bn_scale=True, name=f"{name}_proj")
    outputs = proj(x)
    return tf.keras.Model(inputs, outputs, name=name)


def _pick(value, default):
    """Like `value or default`, but only falls back when value is None (so 0/0.0 are respected)."""
    return default if value is None else value


def build_encoder_for_method(method: str, input_shape=(256, 256, 3), **kwargs) -> tf.keras.Model:
    """Dispatch to the right encoder recipe for a given SSL method (or 'none' for the baseline)."""
    if method == "barlow_twins":
        return build_barlow_encoder(
            input_shape,
            hidden_dim=_pick(kwargs.get("project_dim"), 256),
            weight_decay=_pick(kwargs.get("weight_decay"), 1e-6),
        )
    if method in ("byol", "simclr", "simsiam"):
        return build_pooled_encoder(input_shape, name=f"{method}_encoder")
    if method == "moco":
        return build_moco_encoder(
            input_shape,
            proj_dim=_pick(kwargs.get("project_dim"), 256),
            hidden_dim=_pick(kwargs.get("moco_hidden_dim"), 512),
        )
    if method == "none":
        return build_pooled_encoder(input_shape, name="baseline_encoder")
    raise ValueError(f"Unknown method '{method}'")


def build_segmentation_model(encoder: tf.keras.Model, num_classes: int = 1) -> tf.keras.Model:
    """The shared downstream builder: slice `encoder`'s VGG16 trunk by layer name and attach a decoder.

    Works for every encoder produced above (any extra head layers past block5_conv3 are simply left
    disconnected from the new output graph).
    """
    skip_connections = [encoder.get_layer(n).output for n in SKIP_LAYER_NAMES]
    x = encoder.get_layer(BOTTLENECK_LAYER_NAME).output
    x = SpatialPyramidPoolingFast(x, pool_size=(5, 5, 5))
    x = decoder_branch_normal(x, skip_connections)
    outputs = output_branch(x, num_classes)
    return tf.keras.Model(encoder.input, outputs, name="segmentation_model")
