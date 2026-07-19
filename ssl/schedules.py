"""Warmup + cosine-decay learning-rate schedule shared by every SSL method.

The notebooks re-implemented this same idea three times under slightly different names
(`WarmUpCosine`, `WarmupCosine`) with cosmetic formula differences -- collapsed here into one
implementation, parameterized the same way as the original `WarmUpCosine`.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate_base: float, total_steps: int, warmup_learning_rate: float, warmup_steps: int):
        super().__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("total_steps must be larger or equal to warmup_steps.")

        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError("learning_rate_base must be larger or equal to warmup_learning_rate.")
            slope = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(step < self.warmup_steps, warmup_rate, learning_rate)

        return tf.where(step > self.total_steps, 0.0, learning_rate, name="learning_rate")

    def get_config(self):
        return {
            "learning_rate_base": self.learning_rate_base,
            "total_steps": self.total_steps,
            "warmup_learning_rate": self.warmup_learning_rate,
            "warmup_steps": self.warmup_steps,
        }


def build_warmup_cosine(
    steps_per_epoch: int,
    epochs: int,
    base_lr: float,
    warmup_epoch_fraction: float = 0.1,
    warmup_learning_rate: float = 0.0,
) -> WarmUpCosine:
    steps_per_epoch = max(1, steps_per_epoch)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(epochs * warmup_epoch_fraction) * steps_per_epoch
    return WarmUpCosine(
        learning_rate_base=base_lr,
        total_steps=total_steps,
        warmup_learning_rate=warmup_learning_rate,
        warmup_steps=warmup_steps,
    )
