"""Segmentation losses and metrics shared by every downstream fine-tuning run, ported verbatim
from the notebooks' common "Metrics"/"Loss" sections.
"""
from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K

SMOOTH = 1e-15


def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + K.epsilon()) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + K.epsilon())


def jacard_similarity(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return intersection / union


def jacard_similarity_threshold(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return intersection / union


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def confusion(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.clip_by_value(tf.reshape(y_pred, [-1]), 0, 1)

    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    return precision, recall


def true_positive(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    return (K.sum(y_pos * y_pred_pos) + SMOOTH) / (K.sum(y_pos) + SMOOTH)


def true_negative(y_true, y_pred, smooth: float = 1):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    return (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def jacard_loss(y_true, y_pred):
    return 1 - jacard_similarity(y_true, y_pred)


def ssim_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if len(y_true.shape) == 3:
        y_true = tf.expand_dims(y_true, axis=-1)
    if len(y_pred.shape) == 3:
        y_pred = tf.expand_dims(y_pred, axis=0)
    return 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)


def FocalLoss(y_true, y_pred, alpha: float = 0.26, gamma: float = 2.3):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    inputs = tf.reshape(y_pred, [-1])
    targets = tf.reshape(y_true, [-1])

    bce = tf.keras.losses.binary_crossentropy(targets, inputs)
    bce_exp = tf.exp(-bce)
    return tf.reduce_mean(alpha * tf.pow((1 - bce_exp), gamma) * bce)


def joint_loss1(y_true, y_pred):
    return (FocalLoss(y_true, y_pred) + ssim_loss(y_true, y_pred) + jacard_loss(y_true, y_pred)) / 3


def SimDice(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return (ssim_loss(y_true, y_pred) + dice_loss(y_true, y_pred)) / 2


def hausdorff95(y_true, y_pred):
    """95th-percentile Hausdorff distance between the foreground point sets of y_true/y_pred."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    true_points = tf.where(y_true > 0.5)
    pred_points = tf.where(y_pred > 0.5)

    def compute_valid_distances(a, b):
        a = tf.cast(a, tf.float32)
        b = tf.cast(b, tf.float32)
        distances = tf.norm(tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0), axis=-1)
        return tf.reduce_min(distances, axis=1)

    def compute_distances(a, b):
        a_empty = tf.equal(tf.size(a), 0)
        b_empty = tf.equal(tf.size(b), 0)
        return tf.cond(
            tf.logical_or(a_empty, b_empty),
            lambda: tf.constant([], dtype=tf.float32),
            lambda: compute_valid_distances(a, b),
        )

    d_true_to_pred = compute_distances(true_points, pred_points)
    d_pred_to_true = compute_distances(pred_points, true_points)
    d_all = tf.concat([d_true_to_pred, d_pred_to_true], axis=0)

    hd95 = tf.cond(
        tf.equal(tf.size(d_all), 0),
        lambda: tf.constant(0.0, dtype=tf.float32),
        lambda: tfp.stats.percentile(d_all, 95.0),
    )
    return tf.squeeze(hd95)
