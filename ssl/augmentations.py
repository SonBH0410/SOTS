"""Two-view SSL augmentation pipelines, one per method family, ported verbatim from the notebooks.

Each class exposes `.two_views(image) -> (view1, view2)` for use in a `tf.data` `.map(...)` call.
The four pipelines are genuinely different (not just cosmetic renames of each other), so they're
kept separate rather than forced into one shared class:

- `TwoViewAugment`   -- Barlow Twins: RandomResizedCrop + HFlip(50%), then each of a pool of
                        photometric transforms applied independently at 50% probability.
- `MocoAugment`       -- MoCo: a fixed pipeline of tf.cond-gated transforms (different per-transform
                        probabilities than Barlow's), including three ultrasound-specific artifact
                        transforms (depth attenuation, gaussian shadow, haze) that Barlow's pipeline
                        defines but never wires in.
- `PaperAugment`      -- SimCLR / SimSiam: the standard SimCLR-style pipeline (RandomResizedCrop ->
                        Flip -> ColorJitter(p=0.8) -> Grayscale(p=0.2) -> GaussianBlur(p=0.5)).
- `BYOLAugment`       -- BYOL: same family as PaperAugment but *asymmetric* between the two views
                        (view 1: blur always, no solarize; view 2: rare blur, some solarize), per
                        Grill et al. 2020 Appendix B.
"""
from __future__ import annotations

import tensorflow as tf


# ---------------------------------------------------------------------------
# Barlow Twins
# ---------------------------------------------------------------------------
class TwoViewAugment:
    def __init__(self, image_height: int = 256, image_width: int = 256, image_channels: int = 3):
        self.H, self.W, self.C = image_height, image_width, image_channels
        self.optional_transforms = [
            self.horizontal_flip,
            self.brightness_jitter,
            self.contrast_jitter,
            self.maximize_contrast,
            self.gaussian_blur,
            self.random_grayscale,
            self.gaussian_noise,
            self.cutout,
            self.random_crop_resize,
        ]

    def two_views(self, image):
        return self.apply(image), self.apply(image)

    def apply(self, image):
        image = tf.cast(image, tf.float32)

        seed_rrc = (tf.random.uniform([], maxval=10000, dtype=tf.int32), tf.random.uniform([], maxval=10000, dtype=tf.int32))
        image = self.random_crop_resize(image, seed=seed_rrc)

        if tf.random.uniform(()) < 0.5:
            image = self.horizontal_flip(image)

        for transform_fn in self.optional_transforms:
            if tf.random.uniform(()) < 0.5:
                seed = (tf.random.uniform([], maxval=10000, dtype=tf.int32), tf.random.uniform([], maxval=10000, dtype=tf.int32))
                try:
                    image = transform_fn(image, seed=seed)
                except TypeError:
                    image = transform_fn(image)

        image = tf.clip_by_value(image, 0.0, 1.0)
        image.set_shape([self.H, self.W, self.C])
        return image

    def maximize_contrast(self, image, seed=(1, 1)):
        factor = tf.random.stateless_uniform([], seed=seed, minval=1.2, maxval=1.5)
        return tf.image.adjust_contrast(image, factor)

    def contrast_jitter(self, image, seed=(1, 1)):
        factor = tf.random.stateless_uniform([], seed=seed, minval=0.8, maxval=1.2)
        return tf.image.adjust_contrast(image, factor)

    def brightness_jitter(self, image, seed=(1, 1)):
        delta = tf.random.stateless_uniform([], seed=seed, minval=-0.2, maxval=0.2)
        return tf.image.adjust_brightness(image, delta)

    def horizontal_flip(self, image):
        return tf.image.flip_left_right(image)

    def cutout(self, image, max_size=60, seed=(1, 1)):
        size = tf.random.stateless_uniform([], seed=seed, minval=max_size // 4, maxval=max_size // 2, dtype=tf.int32)
        x = tf.random.stateless_uniform([], seed=(seed[0] + 1, seed[1] + 1), minval=0, maxval=self.W - size, dtype=tf.int32)
        y = tf.random.stateless_uniform([], seed=(seed[0] + 2, seed[1] + 2), minval=0, maxval=self.H - size, dtype=tf.int32)
        mask = tf.ones((size, size, self.C), dtype=image.dtype)
        mask = tf.image.pad_to_bounding_box(mask, y, x, self.H, self.W)
        return tf.where(mask == 1, tf.zeros_like(image), image)

    def gaussian_blur(self, image, seed=(1, 1)):
        sigma = tf.random.stateless_uniform([], seed=seed, minval=0.1, maxval=0.8)
        radius = tf.cast(3.0 * sigma, tf.int32)
        x = tf.range(-radius, radius + 1, dtype=tf.float32)
        gauss = tf.exp(-tf.square(x) / (2.0 * sigma * sigma))
        gauss_kernel = tf.tensordot(gauss, gauss, axes=0)
        gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
        gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
        gauss_kernel = tf.tile(gauss_kernel, [1, 1, self.C, 1])
        image_exp = tf.expand_dims(image, 0)
        blurred = tf.nn.depthwise_conv2d(image_exp, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
        return tf.squeeze(blurred, 0)

    def random_grayscale(self, image, seed=(1, 1)):
        gray = tf.image.rgb_to_grayscale(image)
        image_gray = tf.tile(gray, [1, 1, 3])
        return tf.cond(tf.random.stateless_uniform([], seed=seed) < 0.3, lambda: image_gray, lambda: image)

    def gaussian_noise(self, image, seed=(1, 1)):
        stddev = tf.random.stateless_uniform([], seed=seed, minval=0.01, maxval=0.05)
        noise = tf.random.stateless_normal(tf.shape(image), mean=0.0, stddev=stddev, dtype=image.dtype, seed=(seed[0] + 1, seed[1] + 1))
        return tf.clip_by_value(image + noise, 0.0, 1.0)

    def random_crop_resize(self, image, min_scale=0.3, max_scale=1.0, aspect_ratio_range=(0.75, 1.33), seed=None):
        if seed is None:
            seed = (tf.random.uniform([], dtype=tf.int32), tf.random.uniform([], dtype=tf.int32))
        original_shape = tf.shape(image)
        h, w = tf.cast(original_shape[0], tf.float32), tf.cast(original_shape[1], tf.float32)
        area = h * w
        target_area = tf.random.stateless_uniform([], seed=seed, minval=min_scale, maxval=max_scale) * area
        aspect_ratio = tf.random.stateless_uniform([], seed=(seed[0] + 1, seed[1] + 1), minval=aspect_ratio_range[0], maxval=aspect_ratio_range[1])
        new_w = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
        new_h = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)
        new_w = tf.minimum(new_w, tf.cast(w, tf.int32))
        new_h = tf.minimum(new_h, tf.cast(h, tf.int32))
        offset_w = tf.random.stateless_uniform([], seed=(seed[0] + 2, seed[1] + 2), minval=0, maxval=w - tf.cast(new_w, tf.float32) + 1, dtype=tf.float32)
        offset_h = tf.random.stateless_uniform([], seed=(seed[0] + 3, seed[1] + 3), minval=0, maxval=h - tf.cast(new_h, tf.float32) + 1, dtype=tf.float32)
        crop = tf.image.crop_to_bounding_box(image, tf.cast(offset_h, tf.int32), tf.cast(offset_w, tf.int32), new_h, new_w)
        resized = tf.image.resize(crop, [self.H, self.W])
        return tf.cast(resized, image.dtype)


# ---------------------------------------------------------------------------
# MoCo (includes ultrasound-specific artifact transforms Barlow's pool defines but never enables)
# ---------------------------------------------------------------------------
def _depth_attenuation(img, attenuation_rate=0.5, max_attenuation=0.1):
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    x = tf.linspace(0.0, 1.0, w)
    y = tf.linspace(0.0, 1.0, h)
    xv, yv = tf.meshgrid(x, y)
    distances = tf.sqrt(tf.square(xv - 0.5) + tf.square(yv))
    attenuation_map = (1.0 - max_attenuation) * tf.exp(-attenuation_rate * distances) + max_attenuation
    attenuation_map = tf.expand_dims(attenuation_map, axis=-1)
    return img * attenuation_map


def _gaussian_shadow(img, strength=(0.8, 1.0), sigma_x=(0.005, 0.2), sigma_y=(0.005, 0.2)):
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    x = tf.range(w, dtype=tf.float32)
    y = tf.range(h, dtype=tf.float32)
    xv, yv = tf.meshgrid(x, y)
    mu_x = tf.random.uniform([], 0.0, tf.cast(w, tf.float32))
    mu_y = tf.random.uniform([], 0.0, tf.cast(h, tf.float32))
    strength_val = tf.random.uniform([], *strength)
    sigma_x_val = tf.random.uniform([], *sigma_x) * tf.cast(w, tf.float32)
    sigma_y_val = tf.random.uniform([], *sigma_y) * tf.cast(h, tf.float32)
    gauss = 1.0 - strength_val * tf.exp(
        -(((xv - mu_x) ** 2) / (2.0 * sigma_x_val**2) + ((yv - mu_y) ** 2) / (2.0 * sigma_y_val**2))
    )
    gauss = tf.expand_dims(gauss, axis=-1)
    return img * gauss


def _haze_artifact(img, radius=(0.05, 0.95), sigma=(0, 0.1)):
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    x = tf.linspace(0.0, 1.0, w)
    y = tf.linspace(0.0, 1.0, h)
    xv, yv = tf.meshgrid(x, y)
    r = tf.sqrt(tf.square(xv - 0.5) + tf.square(yv))
    haze_radius = tf.random.uniform([], *radius)
    haze_sigma = tf.random.uniform([], *sigma)
    haze = tf.random.uniform(tf.shape(r)) * tf.exp(-tf.square(r - haze_radius) / (2.0 * haze_sigma**2))
    haze = tf.expand_dims(haze, axis=-1)
    img = img + 0.5 * haze
    return tf.clip_by_value(img, 0.0, 1.0)


def _speckle_reduction(img, sigma=1.0):
    channels = tf.shape(img)[-1]
    radius = tf.cast(3.0 * sigma, tf.int32)
    x = tf.range(-radius, radius + 1, dtype=tf.float32)
    gauss = tf.exp(-0.5 * tf.square(x / sigma))
    gauss_kernel = gauss / tf.reduce_sum(gauss)
    gauss_kernel = tf.reshape(gauss_kernel, [-1, 1, 1])
    gauss_kernel = tf.tile(gauss_kernel, [1, 1, channels])
    img = tf.expand_dims(img, axis=0)
    img = tf.nn.depthwise_conv2d(img, filter=tf.expand_dims(gauss_kernel, -1), strides=[1, 1, 1, 1], padding="SAME")
    img = tf.transpose(img, [0, 2, 1, 3])
    img = tf.nn.depthwise_conv2d(img, filter=tf.expand_dims(gauss_kernel, -1), strides=[1, 1, 1, 1], padding="SAME")
    img = tf.transpose(img, [0, 2, 1, 3])
    return tf.squeeze(img, axis=0)


class MocoAugment:
    def __init__(self, image_height: int = 256, image_width: int = 256, image_channels: int = 3):
        self.H, self.W, self.C = image_height, image_width, image_channels

    def two_views(self, image):
        return self.apply(image), self.apply(image)

    def apply(self, image):
        image = tf.cast(image, tf.float32)

        image = tf.cond(tf.random.uniform(()) < 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
        image = tf.cond(tf.random.uniform(()) < 0.4, lambda: tf.image.adjust_brightness(image, tf.random.uniform([], -0.2, 0.2)), lambda: image)
        image = tf.cond(tf.random.uniform(()) < 0.4, lambda: tf.image.adjust_contrast(image, tf.random.uniform([], 0.8, 1.2)), lambda: image)
        image = tf.cond(tf.random.uniform(()) < 0.3, lambda: tf.image.adjust_contrast(image, tf.random.uniform([], 1.2, 1.5)), lambda: image)
        image = tf.cond(tf.random.uniform(()) < 0.3, lambda: self.gaussian_blur(image), lambda: image)
        image = tf.cond(tf.random.uniform(()) < 0.3, lambda: self.gaussian_noise(image), lambda: image)
        image = tf.cond(tf.random.uniform(()) < 0.3, lambda: self.cutout(image), lambda: image)
        image = tf.cond(tf.random.uniform(()) < 0.3, lambda: self.random_grayscale(image), lambda: image)
        image = tf.cond(tf.random.uniform(()) < 0.3, lambda: _depth_attenuation(image), lambda: image)
        image = tf.cond(tf.random.uniform(()) < 0.3, lambda: _gaussian_shadow(image), lambda: image)
        image = tf.cond(tf.random.uniform(()) < 0.3, lambda: _haze_artifact(image), lambda: image)
        image = tf.cond(tf.random.uniform(()) < 0.3, lambda: _speckle_reduction(image), lambda: image)

        image = tf.clip_by_value(image, 0.0, 1.0)
        image.set_shape([self.H, self.W, self.C])
        return image

    def cutout(self, image, max_size=60):
        size = tf.random.uniform([], max_size // 4, max_size // 2, dtype=tf.int32)
        x = tf.random.uniform([], 0, self.W - size, dtype=tf.int32)
        y = tf.random.uniform([], 0, self.H - size, dtype=tf.int32)
        mask = tf.ones((size, size, self.C), dtype=image.dtype)
        mask = tf.image.pad_to_bounding_box(mask, y, x, self.H, self.W)
        return tf.where(mask == 1, tf.zeros_like(image), image)

    def gaussian_blur(self, image):
        sigma = tf.random.uniform([], 0.1, 0.8)
        radius = tf.cast(3.0 * sigma, tf.int32)
        x = tf.range(-radius, radius + 1, dtype=tf.float32)
        gauss = tf.exp(-tf.square(x) / (2.0 * sigma * sigma))
        gauss_kernel = tf.tensordot(gauss, gauss, axes=0)
        gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
        gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
        gauss_kernel = tf.tile(gauss_kernel, [1, 1, self.C, 1])
        image_exp = tf.expand_dims(image, 0)
        blurred = tf.nn.depthwise_conv2d(image_exp, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
        return tf.squeeze(blurred, 0)

    def random_grayscale(self, image):
        gray = tf.image.rgb_to_grayscale(image)
        image_gray = tf.tile(gray, [1, 1, 3])
        return tf.cond(tf.random.uniform([], 0, 1) < 0.3, lambda: image_gray, lambda: image)

    def gaussian_noise(self, image):
        stddev = tf.random.uniform([], 0.01, 0.05)
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=stddev, dtype=image.dtype)
        return tf.clip_by_value(image + noise, 0.0, 1.0)


# ---------------------------------------------------------------------------
# SimCLR / SimSiam
# ---------------------------------------------------------------------------
class PaperAugment:
    """SimCLR / SimSiam paper augmentation pipeline (Chen et al. 2020 / Chen & He 2021).

    RandomResizedCrop -> Flip -> ColorJitter(p=0.8) -> Grayscale(p=0.2) -> GaussianBlur(p=0.5).
    """

    def __init__(self, image_height: int = 256, image_width: int = 256, s: float = 1.0, blur_p: float = 0.5, jitter_p: float = 0.8, gray_p: float = 0.2):
        self.H, self.W = image_height, image_width
        self.s = s
        self.blur_p, self.jitter_p, self.gray_p = blur_p, jitter_p, gray_p

    def two_views(self, image):
        return self.apply(image), self.apply(image)

    def random_resized_crop(self, image):
        scale = tf.random.uniform([], 0.2, 1.0)
        side = tf.cast(tf.cast(tf.minimum(self.H, self.W), tf.float32) * tf.sqrt(scale), tf.int32)
        side = tf.maximum(side, 16)
        crop = tf.image.random_crop(image, size=[side, side, 3])
        return tf.image.resize(crop, [self.H, self.W])

    def color_jitter(self, image):
        s = self.s
        image = tf.image.random_brightness(image, 0.8 * s)
        image = tf.image.random_contrast(image, 1.0 - 0.8 * s, 1.0 + 0.8 * s)
        image = tf.image.random_saturation(image, 1.0 - 0.8 * s, 1.0 + 0.8 * s)
        image = tf.image.random_hue(image, 0.2 * s)
        return tf.clip_by_value(image, 0.0, 1.0)

    def random_grayscale(self, image):
        gray = tf.image.rgb_to_grayscale(image)
        return tf.tile(gray, [1, 1, 3])

    def gaussian_blur(self, image):
        ksize = tf.cast(tf.cast(self.H, tf.float32) * 0.1, tf.int32)
        ksize = ksize + 1 - ksize % 2
        sigma = tf.random.uniform([], 0.1, 2.0)
        x = tf.cast(tf.range(-(ksize // 2), ksize // 2 + 1), tf.float32)
        g = tf.exp(-(x**2) / (2.0 * sigma**2))
        g = g / tf.reduce_sum(g)
        kx = tf.tile(tf.reshape(g, [1, ksize, 1, 1]), [1, 1, 3, 1])
        ky = tf.tile(tf.reshape(g, [ksize, 1, 1, 1]), [1, 1, 3, 1])
        img = tf.expand_dims(image, 0)
        img = tf.nn.depthwise_conv2d(img, kx, [1, 1, 1, 1], "SAME")
        img = tf.nn.depthwise_conv2d(img, ky, [1, 1, 1, 1], "SAME")
        return tf.squeeze(img, 0)

    def apply(self, image):
        image = self.random_resized_crop(image)
        image = tf.image.random_flip_left_right(image)
        if tf.random.uniform([]) < self.jitter_p:
            image = self.color_jitter(image)
        if tf.random.uniform([]) < self.gray_p:
            image = self.random_grayscale(image)
        if tf.random.uniform([]) < self.blur_p:
            image = self.gaussian_blur(image)
        return image


# ---------------------------------------------------------------------------
# BYOL
# ---------------------------------------------------------------------------
class _BYOLViewAugment:
    def __init__(self, H=256, W=256, blur_p=1.0, solarize_p=0.0, jitter_p=0.8, gray_p=0.2, s=1.0):
        self.H, self.W = H, W
        self.blur_p, self.solarize_p = blur_p, solarize_p
        self.jitter_p, self.gray_p = jitter_p, gray_p
        self.s = s

    def random_resized_crop(self, image):
        scale = tf.random.uniform([], 0.08, 1.0)
        side = tf.cast(tf.cast(tf.minimum(self.H, self.W), tf.float32) * tf.sqrt(scale), tf.int32)
        side = tf.maximum(side, 16)
        crop = tf.image.random_crop(image, size=[side, side, 3])
        return tf.image.resize(crop, [self.H, self.W])

    def color_jitter(self, image):
        s = self.s
        image = tf.image.random_brightness(image, 0.4 * s)
        image = tf.image.random_contrast(image, 1.0 - 0.4 * s, 1.0 + 0.4 * s)
        image = tf.image.random_saturation(image, 1.0 - 0.2 * s, 1.0 + 0.2 * s)
        image = tf.image.random_hue(image, 0.1 * s)
        return tf.clip_by_value(image, 0.0, 1.0)

    def random_grayscale(self, image):
        gray = tf.image.rgb_to_grayscale(image)
        return tf.tile(gray, [1, 1, 3])

    def gaussian_blur(self, image):
        ksize = tf.cast(tf.cast(self.H, tf.float32) * 0.1, tf.int32)
        ksize = ksize + 1 - ksize % 2
        sigma = tf.random.uniform([], 0.1, 2.0)
        x = tf.cast(tf.range(-(ksize // 2), ksize // 2 + 1), tf.float32)
        g = tf.exp(-(x**2) / (2.0 * sigma**2))
        g = g / tf.reduce_sum(g)
        kx = tf.tile(tf.reshape(g, [1, ksize, 1, 1]), [1, 1, 3, 1])
        ky = tf.tile(tf.reshape(g, [ksize, 1, 1, 1]), [1, 1, 3, 1])
        img = tf.expand_dims(image, 0)
        img = tf.nn.depthwise_conv2d(img, kx, [1, 1, 1, 1], "SAME")
        img = tf.nn.depthwise_conv2d(img, ky, [1, 1, 1, 1], "SAME")
        return tf.squeeze(img, 0)

    def solarize(self, image, threshold=0.5):
        return tf.where(image >= threshold, 1.0 - image, image)

    def apply(self, image):
        image = self.random_resized_crop(image)
        image = tf.image.random_flip_left_right(image)
        if tf.random.uniform([]) < self.jitter_p:
            image = self.color_jitter(image)
        if tf.random.uniform([]) < self.gray_p:
            image = self.random_grayscale(image)
        if tf.random.uniform([]) < self.blur_p:
            image = self.gaussian_blur(image)
        if tf.random.uniform([]) < self.solarize_p:
            image = self.solarize(image)
        return image


class BYOLAugment:
    """Asymmetric T / T' views, per Grill et al. 2020 Appendix B."""

    def __init__(self, image_height: int = 256, image_width: int = 256):
        self.view1 = _BYOLViewAugment(image_height, image_width, blur_p=1.0, solarize_p=0.0)
        self.view2 = _BYOLViewAugment(image_height, image_width, blur_p=0.1, solarize_p=0.2)

    def two_views(self, image):
        return self.view1.apply(image), self.view2.apply(image)
