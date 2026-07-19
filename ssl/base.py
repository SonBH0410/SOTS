"""Shared base class for every SSL method: a loss tracker plus optional gradient accumulation.

Only Barlow Twins' notebook actually used gradient accumulation (accum_steps=4, to reach an
effective batch size of 16 on a batch_size=4 pipeline). Lifting that logic into a shared base class
means `--accum-steps` works consistently for every method instead of only one.
"""
from __future__ import annotations

import tensorflow as tf


class SSLModel(tf.keras.Model):
    def __init__(self, accum_steps: int = 1, name: str | None = None):
        super().__init__(name=name)
        self.accum_steps = max(1, int(accum_steps))
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self._grad_accumulator = None
        self._accum_step_counter = None

    @property
    def metrics(self):
        return [self.loss_tracker]

    def accumulate_and_apply(self, tape: tf.GradientTape, loss: tf.Tensor, trainable_vars: list) -> None:
        """Apply gradients directly (accum_steps=1), or accumulate over `accum_steps` batches."""
        if self.accum_steps <= 1:
            grads = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(grads, trainable_vars))
            return

        if self._grad_accumulator is None:
            self._grad_accumulator = [tf.Variable(tf.zeros_like(v), trainable=False) for v in trainable_vars]
        if self._accum_step_counter is None:
            self._accum_step_counter = tf.Variable(0, dtype=tf.int32, trainable=False)

        self._accum_step_counter.assign_add(1)

        scaled_loss = loss / tf.cast(self.accum_steps, loss.dtype)
        scaled_grads = tape.gradient(scaled_loss, trainable_vars)
        for accumulator, grad in zip(self._grad_accumulator, scaled_grads):
            accumulator.assign_add(grad)

        def apply_and_reset():
            grads_to_apply = [tf.identity(g) for g in self._grad_accumulator]
            with tf.control_dependencies(grads_to_apply):
                for i, v in enumerate(trainable_vars):
                    self._grad_accumulator[i].assign(tf.zeros_like(v))
            return grads_to_apply

        def zeros():
            return [tf.zeros_like(v) for v in trainable_vars]

        final_grads = tf.cond(
            tf.equal(self._accum_step_counter % self.accum_steps, 0),
            true_fn=apply_and_reset,
            false_fn=zeros,
        )
        self.optimizer.apply_gradients(zip(final_grads, trainable_vars))
