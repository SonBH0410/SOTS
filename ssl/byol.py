"""BYOL (Grill et al. 2020).

Online network: encoder -> projector -> predictor.
Target network: encoder -> projector (no predictor), EMA-updated, never receives gradients.
Loss: symmetrized normalized-L2 (2 - 2*cos_sim) regression between predictor output and the
stop-gradiented target projection of the other view.
"""
from __future__ import annotations

import tensorflow as tf

from ..models.heads import mlp_head
from .base import SSLModel


def byol_regression_loss(p: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    return 2.0 - 2.0 * tf.reduce_sum(p * z, axis=1)


class BYOL(SSLModel):
    def __init__(
        self,
        encoder: tf.keras.Model,
        proj_hidden_dim: int = 4096,
        proj_out_dim: int = 256,
        pred_hidden_dim: int = 4096,
        base_momentum: float = 0.996,
        total_steps: int | None = None,
        accum_steps: int = 1,
        name: str = "byol",
    ):
        super().__init__(accum_steps=accum_steps, name=name)

        feat_dim = encoder.output_shape[-1]

        self.encoder = encoder
        self.projector = mlp_head(feat_dim, [proj_hidden_dim], proj_out_dim, final_bn=False, name="online_projector")
        self.predictor = mlp_head(proj_out_dim, [pred_hidden_dim], proj_out_dim, final_bn=False, name="predictor")

        self.target_encoder = tf.keras.models.clone_model(encoder)
        self.target_projector = mlp_head(feat_dim, [proj_hidden_dim], proj_out_dim, final_bn=False, name="target_projector")
        self.target_encoder.set_weights(self.encoder.get_weights())
        self.target_projector.set_weights(self.projector.get_weights())
        self.target_encoder.trainable = False
        self.target_projector.trainable = False

        self.base_momentum = base_momentum
        self.total_steps = float(total_steps) if total_steps else None
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        self.tau_tracker = tf.keras.metrics.Mean(name="tau")

    @property
    def metrics(self):
        return [self.loss_tracker, self.tau_tracker]

    def _current_tau(self) -> tf.Tensor:
        if self.total_steps is None:
            return tf.constant(self.base_momentum, dtype=tf.float32)
        step = tf.cast(self.global_step, tf.float32)
        cos = tf.cos(tf.constant(3.14159265, tf.float32) * step / self.total_steps)
        return 1.0 - (1.0 - self.base_momentum) * (cos + 1.0) / 2.0

    def _ema_update(self, tau: tf.Tensor) -> None:
        for w_online, w_target in zip(self.encoder.trainable_variables, self.target_encoder.trainable_variables):
            w_target.assign(tau * w_target + (1.0 - tau) * w_online)
        for w_online, w_target in zip(self.projector.trainable_variables, self.target_projector.trainable_variables):
            w_target.assign(tau * w_target + (1.0 - tau) * w_online)

    def call(self, x, training=False):
        h = self.encoder(x, training=training)
        z = self.projector(h, training=training)
        p = self.predictor(z, training=training)
        return p, z

    def train_step(self, data):
        x1, x2 = data

        t1 = tf.stop_gradient(self.target_projector(self.target_encoder(x1, training=False), training=False))
        t2 = tf.stop_gradient(self.target_projector(self.target_encoder(x2, training=False), training=False))

        with tf.GradientTape() as tape:
            z1 = self.projector(self.encoder(x1, training=True), training=True)
            z2 = self.projector(self.encoder(x2, training=True), training=True)
            p1 = self.predictor(z1, training=True)
            p2 = self.predictor(z2, training=True)

            loss = 0.5 * tf.reduce_mean(byol_regression_loss(p1, t2) + byol_regression_loss(p2, t1))

        online_vars = self.encoder.trainable_variables + self.projector.trainable_variables + self.predictor.trainable_variables
        self.accumulate_and_apply(tape, loss, online_vars)

        tau = self._current_tau()
        self._ema_update(tau)
        self.global_step.assign_add(1)

        self.loss_tracker.update_state(loss)
        self.tau_tracker.update_state(tau)
        return {"loss": self.loss_tracker.result(), "tau": self.tau_tracker.result()}
