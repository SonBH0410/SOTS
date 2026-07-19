"""Barlow Twins (Zbontar et al. 2021).

Ported from the variant that was actually trained in the notebooks (`BarlowTwins_1_Accumulate`,
using the plain symmetric 2-view cross-correlation loss + gradient accumulation) -- not the two
sibling classes that were built but never fit (a top-k multi-view loss variant, and a
`MirroredStrategy` experiment).
"""
from __future__ import annotations

import tensorflow as tf

from .base import SSLModel


def off_diagonal(mat: tf.Tensor) -> tf.Tensor:
    n = tf.shape(mat)[0]
    flat = tf.reshape(mat, [-1])[:-1]
    off = tf.reshape(flat, (n - 1, n + 1))[:, 1:]
    return tf.reshape(off, [-1])


def normalize_batch(z: tf.Tensor) -> tf.Tensor:
    mean = tf.reduce_mean(z, axis=0, keepdims=True)
    std = tf.math.reduce_std(z, axis=0, keepdims=True) + 1e-3
    return (z - mean) / std


def compute_barlow_twins_loss(z1: tf.Tensor, z2: tf.Tensor, lambd: float = 5e-3) -> tf.Tensor:
    """Standard Barlow Twins cross-correlation loss between two (B, D) embedding batches."""
    batch_size = tf.cast(tf.shape(z1)[0], z1.dtype)

    z1_norm = normalize_batch(z1)
    z2_norm = normalize_batch(z2)

    c = tf.matmul(z1_norm, z2_norm, transpose_a=True) / batch_size

    on_diag = tf.linalg.diag_part(c)
    on_diag_loss = tf.reduce_sum(tf.square(on_diag - 1.0))
    off_diag_loss = tf.reduce_sum(tf.square(off_diagonal(c)))

    return on_diag_loss + lambd * off_diag_loss


class BarlowTwins(SSLModel):
    def __init__(self, encoder: tf.keras.Model, lambd: float = 5e-3, accum_steps: int = 4, name: str = "barlow_twins"):
        super().__init__(accum_steps=accum_steps, name=name)
        self.encoder = encoder
        self.lambd = lambd

    def call(self, inputs, training=False):
        return self.encoder(inputs, training=training)

    def train_step(self, data):
        view1, view2 = data
        with tf.GradientTape() as tape:
            z1 = self.encoder(view1, training=True)
            z2 = self.encoder(view2, training=True)
            loss = compute_barlow_twins_loss(z1, z2, self.lambd)

        self.accumulate_and_apply(tape, loss, self.encoder.trainable_variables)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
