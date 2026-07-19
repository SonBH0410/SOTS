"""SimSiam (Chen & He, 2021). Single shared encoder, predictor on one branch, stop-gradient loss."""
from __future__ import annotations

import tensorflow as tf

from ..models.heads import mlp_head
from .base import SSLModel


def simsiam_loss(p: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
    """Negative cosine similarity between predictor output p and stop-gradiented projection z."""
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    return -tf.reduce_mean(tf.reduce_sum(p * z, axis=1))


class SimSiam(SSLModel):
    def __init__(
        self,
        encoder: tf.keras.Model,
        proj_hidden_dim: int = 2048,
        proj_out_dim: int = 2048,
        pred_hidden_dim: int = 512,
        accum_steps: int = 1,
        name: str = "simsiam",
    ):
        super().__init__(accum_steps=accum_steps, name=name)
        self.encoder = encoder
        feat_dim = encoder.output_shape[-1]
        self.projection = mlp_head(
            feat_dim, [proj_hidden_dim, proj_hidden_dim], proj_out_dim, final_bn=True, final_bn_scale=False, name="proj"
        )
        self.predictor = mlp_head(proj_out_dim, [pred_hidden_dim], proj_out_dim, final_bn=False, name="pred")

    def call(self, x, training=False):
        h = self.encoder(x, training=training)
        z = self.projection(h, training=training)
        p = self.predictor(z, training=training)
        return p, z

    def train_step(self, data):
        x1, x2 = data
        with tf.GradientTape() as tape:
            z1 = self.projection(self.encoder(x1, training=True), training=True)
            z2 = self.projection(self.encoder(x2, training=True), training=True)
            p1 = self.predictor(z1, training=True)
            p2 = self.predictor(z2, training=True)

            loss = 0.5 * (simsiam_loss(p1, z2) + simsiam_loss(p2, z1))

        self.accumulate_and_apply(tape, loss, self.trainable_variables)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
