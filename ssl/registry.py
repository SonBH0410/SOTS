"""Method name -> augmentation / dataset / model / hyperparameter dispatch.

Per-method defaults below are taken directly from the notebooks' actual (not merely commented-intent)
call sites -- e.g. SimSiam really was compiled with plain eager Adam(1e-4) despite an in-notebook
comment describing an SGD+cosine "paper" config that was built but never used.
"""
from __future__ import annotations

import tensorflow as tf

from ..models.encoders import build_encoder_for_method
from .augmentations import BYOLAugment, MocoAugment, PaperAugment, TwoViewAugment
from .barlow_twins import BarlowTwins
from .byol import BYOL
from .moco import MoCo
from .simclr import SimCLR
from .simsiam import SimSiam
from .method_names import METHODS

# lr: fixed base learning rate, or None if it's derived from batch size via lr_scale_per_256.
# warmup_epochs: fixed epoch count, or None if derived from epochs via warmup_epoch_fraction.
DEFAULT_HYPERPARAMS = {
    "barlow_twins": dict(
        batch_size=4, epochs=200, lr=1e-3, lr_scale_per_256=None,
        warmup_epochs=None, warmup_epoch_fraction=0.1,
        optimizer="sgd", use_schedule=True, run_eagerly=False,
        project_dim=256, lambd=5e-3, accum_steps=4,
    ),
    "byol": dict(
        batch_size=8, epochs=200, lr=None, lr_scale_per_256=0.2,
        warmup_epochs=10, warmup_epoch_fraction=None,
        optimizer="sgd", use_schedule=True, run_eagerly=False,
        project_dim=256, momentum=0.996, accum_steps=1,
    ),
    "moco": dict(
        batch_size=8, epochs=200, lr=1e-4, lr_scale_per_256=None,
        warmup_epochs=None, warmup_epoch_fraction=0.1,
        optimizer="adam", use_schedule=True, run_eagerly=True,
        project_dim=256, momentum=0.999, temperature=0.07, queue_size=4096, accum_steps=1,
    ),
    "simclr": dict(
        batch_size=8, epochs=200, lr=None, lr_scale_per_256=0.3,
        warmup_epochs=10, warmup_epoch_fraction=None,
        optimizer="sgd", use_schedule=True, run_eagerly=False,
        project_dim=128, temperature=0.5, accum_steps=1,
    ),
    "simsiam": dict(
        batch_size=8, epochs=200, lr=1e-4, lr_scale_per_256=None,
        warmup_epochs=0, warmup_epoch_fraction=None,
        optimizer="adam", use_schedule=False, run_eagerly=True,
        project_dim=2048, accum_steps=1,
    ),
}


def default_hyperparams(method: str) -> dict:
    if method not in DEFAULT_HYPERPARAMS:
        raise ValueError(f"Unknown SSL method '{method}'. Choices: {METHODS}")
    return dict(DEFAULT_HYPERPARAMS[method])


def _pick(value, default):
    """Like `value or default`, but only falls back when value is None (so 0/0.0 are respected)."""
    return default if value is None else value


def resolve_base_lr(method: str, hp: dict, lr_override: float | None, batch_size: int) -> float:
    if lr_override is not None:
        return lr_override
    if hp["lr"] is not None:
        return hp["lr"]
    return hp["lr_scale_per_256"] * batch_size / 256.0


def resolve_warmup_epochs(hp: dict, epochs: int) -> int:
    if hp["warmup_epochs"] is not None:
        return hp["warmup_epochs"]
    return int(epochs * hp["warmup_epoch_fraction"])


def build_optimizer(method: str, learning_rate) -> tf.keras.optimizers.Optimizer:
    kind = DEFAULT_HYPERPARAMS[method]["optimizer"]
    if kind == "sgd":
        clipnorm = 1.0 if method == "barlow_twins" else None
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, clipnorm=clipnorm)
    if kind == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    raise ValueError(f"Unknown optimizer kind '{kind}'")


def build_augmenter(method: str, img_size: int = 256):
    if method == "barlow_twins":
        return TwoViewAugment(img_size, img_size)
    if method == "moco":
        return MocoAugment(img_size, img_size)
    if method in ("simclr", "simsiam"):
        return PaperAugment(img_size, img_size)
    if method == "byol":
        return BYOLAugment(img_size, img_size)
    raise ValueError(f"Unknown SSL method '{method}'. Choices: {METHODS}")


def build_ssl_dataset(method: str, images, batch_size: int, img_size: int = 256, seed: int = 42) -> tf.data.Dataset:
    augmenter = build_augmenter(method, img_size)
    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.shuffle(max(1, len(images)), seed=seed)
    ds = ds.map(lambda x: augmenter.two_views(x), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_ssl_model(method: str, input_shape=(256, 256, 3), accum_steps: int = 1, **kwargs) -> tf.keras.Model:
    """kwargs (all optional, fall back to DEFAULT_HYPERPARAMS): project_dim, lambd, momentum,
    temperature, queue_size, weight_decay, total_steps (BYOL EMA schedule length)."""
    hp = default_hyperparams(method)

    if method == "barlow_twins":
        encoder = build_encoder_for_method(
            method,
            input_shape,
            project_dim=_pick(kwargs.get("project_dim"), hp["project_dim"]),
            weight_decay=kwargs.get("weight_decay"),
        )
        return BarlowTwins(encoder, lambd=_pick(kwargs.get("lambd"), hp["lambd"]), accum_steps=accum_steps)

    if method == "byol":
        encoder = build_encoder_for_method(method, input_shape)
        return BYOL(
            encoder,
            proj_out_dim=_pick(kwargs.get("project_dim"), hp["project_dim"]),
            base_momentum=_pick(kwargs.get("momentum"), hp["momentum"]),
            total_steps=kwargs.get("total_steps"),
            accum_steps=accum_steps,
        )

    if method == "moco":
        proj_dim = _pick(kwargs.get("project_dim"), hp["project_dim"])
        encoder_q = build_encoder_for_method(method, input_shape, project_dim=proj_dim)
        encoder_k = build_encoder_for_method(method, input_shape, project_dim=proj_dim)
        return MoCo(
            encoder_q,
            encoder_k,
            proj_dim=proj_dim,
            queue_size=_pick(kwargs.get("queue_size"), hp["queue_size"]),
            momentum=_pick(kwargs.get("momentum"), hp["momentum"]),
            temperature=_pick(kwargs.get("temperature"), hp["temperature"]),
            accum_steps=accum_steps,
        )

    if method == "simclr":
        encoder = build_encoder_for_method(method, input_shape)
        return SimCLR(
            encoder,
            proj_out_dim=_pick(kwargs.get("project_dim"), hp["project_dim"]),
            temperature=_pick(kwargs.get("temperature"), hp["temperature"]),
            accum_steps=accum_steps,
        )

    if method == "simsiam":
        encoder = build_encoder_for_method(method, input_shape)
        return SimSiam(encoder, proj_out_dim=_pick(kwargs.get("project_dim"), hp["project_dim"]), accum_steps=accum_steps)

    raise ValueError(f"Unknown SSL method '{method}'. Choices: {METHODS}")


def get_pretrained_encoder(ssl_model: tf.keras.Model, method: str) -> tf.keras.Model:
    """The sub-model whose weights represent the trained encoder (query encoder for MoCo)."""
    if method == "moco":
        return ssl_model.encoder_q
    return ssl_model.encoder
