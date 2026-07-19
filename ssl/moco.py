"""MoCo v2 (He et al. 2020 / Chen et al. 2020).

Query encoder is trained by gradient descent; key encoder is an EMA copy, never receives
gradients. A CPU-resident circular-buffer queue supplies negatives for the InfoNCE loss.
"""
from __future__ import annotations

import tensorflow as tf

from .base import SSLModel


def moco_loss(q: tf.Tensor, k: tf.Tensor, queue: tf.Tensor, temperature: float) -> tf.Tensor:
    """q, k: (B, D) L2-normalized embeddings. queue: (D, K) L2-normalized negative keys."""
    l_pos = tf.reduce_sum(q * k, axis=1, keepdims=True)  # (B, 1)
    l_neg = tf.matmul(q, queue)  # (B, K)

    logits = tf.concat([l_pos, l_neg], axis=1) / temperature
    labels = tf.zeros((tf.shape(logits)[0],), dtype=tf.int32)  # positive is always column 0
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))


class MoCo(SSLModel):
    def __init__(
        self,
        encoder_q: tf.keras.Model,
        encoder_k: tf.keras.Model,
        proj_dim: int = 256,
        queue_size: int = 4096,
        momentum: float = 0.999,
        temperature: float = 0.07,
        accum_steps: int = 1,
        name: str = "moco",
    ):
        super().__init__(accum_steps=accum_steps, name=name)

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.encoder_k.trainable = False
        for w_q, w_k in zip(self.encoder_q.weights, self.encoder_k.weights):
            w_k.assign(w_q)

        self.proj_dim = proj_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        with tf.device("/CPU:0"):
            self.queue = tf.Variable(tf.random.normal((proj_dim, queue_size)), trainable=False, name="moco_queue")
            self.queue.assign(tf.math.l2_normalize(self.queue, axis=0))
            self.queue_ptr = tf.Variable(0, dtype=tf.int32, trainable=False, name="queue_ptr")

    def call(self, inputs, training=False):
        return self.encoder_q(inputs, training=training)

    def _momentum_update(self) -> None:
        for w_q, w_k in zip(self.encoder_q.weights, self.encoder_k.weights):
            w_k.assign(self.momentum * w_k + (1.0 - self.momentum) * w_q)

    def _dequeue_and_enqueue(self, keys: tf.Tensor) -> None:
        batch_size = tf.shape(keys)[0]
        ptr = self.queue_ptr
        idx = tf.expand_dims(tf.range(ptr, ptr + batch_size) % self.queue_size, axis=1)
        new_queue = tf.tensor_scatter_nd_update(tf.transpose(self.queue), idx, keys)
        self.queue.assign(tf.transpose(new_queue))
        self.queue_ptr.assign((ptr + batch_size) % self.queue_size)

    def train_step(self, data):
        x_q, x_k = data

        with tf.GradientTape() as tape:
            q = tf.math.l2_normalize(self.encoder_q(x_q, training=True), axis=1)
            k = tf.stop_gradient(tf.math.l2_normalize(self.encoder_k(x_k, training=False), axis=1))

            with tf.device("/CPU:0"):
                q_cpu = tf.identity(q)
                k_cpu = tf.identity(k)
                loss = moco_loss(q_cpu, k_cpu, self.queue.read_value(), self.temperature)

        self.accumulate_and_apply(tape, loss, self.encoder_q.trainable_variables)

        self._momentum_update()
        with tf.device("/CPU:0"):
            self._dequeue_and_enqueue(k_cpu)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
