"""Shared VGG16-U-Net building blocks, ported verbatim from the "Model" section common to all 15 notebooks."""
from __future__ import annotations

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG16


def conv_block(x, filters, kernel_size=3, padding="same", activation="relu"):
    x = L.Conv2D(filters, kernel_size, padding=padding)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation(activation)(x)

    x = L.Conv2D(filters, kernel_size, padding=padding)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation(activation)(x)
    return x


def encoder_block(x, filters, pool_size=(2, 2), dropout_rate=False):
    x = conv_block(x, filters)
    p = L.MaxPooling2D(pool_size)(x)
    if dropout_rate:
        p = L.Dropout(0.1)(p)
    return x, p


def decoder_block_normal(
    x, skip_connection, filters, kernel_size=3, upsample_size=(2, 2), dropout_rate=False, interpolation="bilinear"
):
    x = L.UpSampling2D(upsample_size, interpolation=interpolation)(x)
    x = L.Concatenate()([x, skip_connection])
    x = conv_block(x, filters)
    if dropout_rate:
        x = L.Dropout(0.1)(x)
    return x


def AttentionGate(g, s, num_filters):
    Wg = L.Conv2D(num_filters, kernel_size=1, strides=1, padding="same")(g)
    Wg = L.BatchNormalization()(Wg)

    Ws = L.Conv2D(num_filters, kernel_size=1, strides=1, padding="same")(s)
    Ws = L.BatchNormalization()(Ws)

    f = L.Activation("relu")(Wg + Ws)
    psi = L.Conv2D(num_filters, kernel_size=1, strides=1, padding="same")(f)
    psi = L.Activation("sigmoid")(psi)

    out = L.Multiply()([s, psi])
    return out


def SpatialPyramidPoolingFast(x, pool_size=(5, 5, 5)):
    _, _, _, channels = K.int_shape(x)
    conv1 = L.Conv2D(channels // 2, kernel_size=1, strides=1, padding="same")(x)
    conv1 = L.BatchNormalization()(conv1)
    conv1 = L.Activation("swish")(conv1)

    pool1 = L.MaxPooling2D(pool_size=(pool_size[0], pool_size[0]), strides=(1, 1), padding="same")(conv1)
    pool2 = L.MaxPooling2D(pool_size=(pool_size[1], pool_size[1]), strides=(1, 1), padding="same")(pool1)
    pool3 = L.MaxPooling2D(pool_size=(pool_size[2], pool_size[2]), strides=(1, 1), padding="same")(pool2)

    comb = L.Concatenate(axis=-1)([conv1, pool1, pool2, pool3])

    out = L.Conv2D(channels, kernel_size=1, strides=1, padding="same")(comb)
    out = L.BatchNormalization()(out)
    out = L.Activation("swish")(out)
    return out


def VGG16_encoder_branch(inputs):
    """VGG16 (ImageNet weights) as the shared conv trunk, with named skip taps.

    Every SSL method's encoder, and the shared segmentation-model builder, rely on the
    resulting layer names (block1_conv2 .. block5_conv3) being present regardless of what
    head gets appended after this trunk.
    """
    base_model = VGG16(weights="imagenet", include_top=False, input_tensor=inputs)
    skip_connections = [
        base_model.get_layer("block1_conv2").output,  # (B, 256, 256, 64)
        base_model.get_layer("block2_conv2").output,  # (B, 128, 128, 128)
        base_model.get_layer("block3_conv3").output,  # (B, 64, 64, 256)
        base_model.get_layer("block4_conv3").output,  # (B, 32, 32, 512)
    ]
    x = base_model.get_layer("block5_conv3").output  # (B, 16, 16, 512)
    return x, skip_connections


def bottle_neck(inputs, dropout=False):
    x = conv_block(inputs, 1024)
    if dropout:
        x = L.Dropout(0.1)(x)
    return x


def decoder_branch_normal(x, skip_connections):
    num_filters = [512, 256, 128, 64]
    for i, f in enumerate(num_filters):
        x = decoder_block_normal(x, skip_connections[-(i + 1)], f)
    return x


def output_branch(x, num_classes):
    x = L.Conv2D(num_classes, kernel_size=1, padding="same")(x)
    x = L.Activation("sigmoid")(x)
    return x
