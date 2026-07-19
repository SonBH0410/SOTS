"""SimCLR (Chen et al. 2020). Single shared encoder, no momentum network, NT-Xent loss."""
from __future__ import annotations

import tensorflow as tf

from ..models.heads import mlp_head
from .base import SSLModel


def nt_xent_loss(z1: tf.Tensor, z2: tf.Tensor, temperature: float = 0.5) -> tf.Tensor:
    """Normalized Temperature-scaled Cross-Entropy loss over two (B, D) view embeddings."""
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)

    batch_size = tf.shape(z1)[0]
    z = tf.concat([z1, z2], axis=0)  # (2B, D)

    sim_matrix = tf.matmul(z, z, transpose_b=True) / temperature  # (2B, 2B)
    self_mask = tf.eye(2 * batch_size, dtype=tf.bool)
    sim_matrix = tf.where(self_mask, tf.fill(tf.shape(sim_matrix), -1e9), sim_matrix)

    pos_sim_1 = tf.reduce_sum(z1 * z2, axis=1) / temperature
    pos_sim_2 = tf.reduce_sum(z2 * z1, axis=1) / temperature
    positives = tf.concat([pos_sim_1, pos_sim_2], axis=0)  # (2B,)

    neg_logits = tf.math.reduce_logsumexp(sim_matrix, axis=1)  # (2B,)
    return tf.reduce_mean(-positives + neg_logits)


class SimCLR(SSLModel):
    def __init__(
        self,
        encoder: tf.keras.Model,
        proj_hidden_dim: int = 2048,
        proj_out_dim: int = 128,
        temperature: float = 0.5,
        accum_steps: int = 1,
        name: str = "simclr",
    ):
        super().__init__(accum_steps=accum_steps, name=name)
        self.encoder = encoder
        self.temperature = temperature
        feat_dim = encoder.output_shape[-1]
        self.projection = mlp_head(feat_dim, [proj_hidden_dim], proj_out_dim, final_bn=False, name="proj")

    def call(self, x, training=False):
        return self.projection(self.encoder(x, training=training), training=training)

    def train_step(self, data):
        x1, x2 = data
        with tf.GradientTape() as tape:
            z1 = self.projection(self.encoder(x1, training=True), training=True)
            z2 = self.projection(self.encoder(x2, training=True), training=True)
            loss = nt_xent_loss(z1, z2, self.temperature)

        self.accumulate_and_apply(tape, loss, self.trainable_variables)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
