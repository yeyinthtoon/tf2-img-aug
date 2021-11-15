"""Implemenations of frequently use image augmentation methods"""
from functools import partial
from typing import Optional, Tuple, Union

import tensorflow as tf


def _make_pairs(batch_size: int, threshold: float) -> tf.Tensor:
    """Generate pairs from given batch size.

    Args:
        batch_size (int) : Batch Size.
        threshold (float) : threshold to generate pairs with different index.

    Returns:
        tf.Tensor : Pairs of index for given batch size.
    """
    image_prob_grib = tf.fill((batch_size, batch_size), 0.5, name=None)
    image_log_prob_grib = tf.math.log(tf.linalg.set_diag(image_prob_grib, tf.zeros(batch_size) + 1e-7))
    mix_indices = tf.squeeze(tf.random.categorical(image_log_prob_grib, 1, dtype=tf.int32))
    ranges = tf.range(batch_size)
    # pylint: disable=no-value-for-parameter
    mix_indices = tf.where(tf.random.uniform([batch_size]) > threshold, mix_indices, ranges)
    pairs = tf.transpose([ranges, mix_indices])
    return pairs


def _estimated_beta_distribution(
    concentration_0: float = 1.0, concentration_1: float = 1.0, size: Optional[int] = None
) -> tf.Tensor:
    """Beta distribution based on gamma.

    Args:
        concentration_0 (float, optional): Alpaha, positive (>0). Defaults to 1.0.
        concentration_1 (float, optional): Beta, positive (>0). Defaults to 1.0.
        size (int, optional): Default to None.

    Returns:
        tf.Tensor: Estimated beta distribution based on Alpha and Beta.
    """
    if isinstance(size, type(None)):
        x = tf.random.gamma(shape=[], alpha=concentration_0)
        y = tf.random.gamma(shape=[], alpha=concentration_1)
    else:
        x = tf.random.gamma(shape=[size], alpha=concentration_0)
        y = tf.random.gamma(shape=[size], alpha=concentration_1)
    return x / (x + y)


def _get_random_box(
    combination_ratio: float, height: int, width: int
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """random bounding box generator.

    Args:
        combination_ratio (float): target ratio of bounding box
        height (int): height of image
        width (int): width of image

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: Tuple[
            Vertical coordinate of the top-left corner, Horizontal coordinate of the top-left corner, Height, Width]
    """
    cut_rat = tf.math.sqrt(1.0 - combination_ratio)
    cut_w = tf.cast((width * cut_rat), tf.int32)
    cut_h = tf.cast((height * cut_rat), tf.int32)
    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    cut_x = tf.random.uniform([], minval=0, maxval=width, dtype=tf.int32)
    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    cut_y = tf.random.uniform([], minval=0, maxval=height, dtype=tf.int32)

    x2 = tf.clip_by_value(cut_x + cut_w // 2, 1, width)
    y2 = tf.clip_by_value(cut_y + cut_h // 2, 1, height)
    x1 = tf.clip_by_value(cut_x - cut_w // 2, 0, x2 - 1)
    y1 = tf.clip_by_value(cut_y - cut_h // 2, 0, y2 - 1)

    width = x2 - x1
    height = y2 - y1

    return x1, y1, height, width


def cutmix(
    images: tf.Tensor,
    labels: tf.Tensor,
    cutmix_threshold: float = 0.5,
    image_dimension: int = 3,
    beta: float = 1.0,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """implementation of cutmix augmentation.
    See [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features]
    (https://arxiv.org/abs/1905.04899)

    Args:
        images (tf.Tensor): Batch of images to process.
        labels (tf.Tensor): Labels of images to process. This function excepted labels to be one-hot encoded.
        cutmix_threshold (float, optional): Defaults to 0.5.
        image_dimension (int, optional): Defaults to 3.
        beta (float, optional): Positive (>0), Defaults to 1.0.

    Raises:
        ValueError: if dimesion of images is less than image_dimension+1.

    Returns:
        Tuple[tf.Tensor,tf.Tensor]: Tuple[images, labels]
    """

    shape = images.shape
    label_shape = labels.shape
    batch_size = tf.shape(images)[0]
    if labels.dtype != tf.float32:
        labels = tf.cast(labels, tf.float32)

    if beta <= 0:
        raise ValueError("Beta must be greater than zero.")
    if images.shape.ndims != image_dimension + 1:
        raise ValueError(f"Only batch of images is accepted. Input Images Dimesnion must be {image_dimension+1}")

    def _mix(pair, height, width):
        img1 = images[pair[0]]
        label1 = labels[pair[0]]
        if pair[0] != pair[1]:
            img2 = images[pair[1]]
            label2 = labels[pair[1]]

            combination_ratio = _estimated_beta_distribution(beta, beta)
            x1, y1, target_height, target_width = _get_random_box(combination_ratio, height, width)

            crop_blank_img1 = tf.image.pad_to_bounding_box(
                tf.image.crop_to_bounding_box(img1, y1, x1, target_height, target_width), y1, x1, height, width
            )
            cropped_img1 = img1 - crop_blank_img1
            crop_blank_img2 = tf.image.pad_to_bounding_box(
                tf.image.crop_to_bounding_box(img2, y1, x1, target_height, target_width), y1, x1, height, width
            )
            mix_img = cropped_img1 + crop_blank_img2

            combination_ratio = 1 - (target_height * target_width) / (height * width)
            combination_ratio = tf.cast(combination_ratio, tf.float32)
            label = combination_ratio * label1 + (1 - combination_ratio) * label2

            return (mix_img, label)
        return (img1, label1)

    pairs = _make_pairs(batch_size, cutmix_threshold)
    mix = partial(_mix, height=shape[1], width=shape[2])
    return tf.map_fn(
        mix,
        pairs,
        fn_output_signature=(
            tf.TensorSpec(shape=shape[1:], dtype=images.dtype),
            tf.TensorSpec(shape=label_shape[1:], dtype=labels.dtype),
        ),
    )


def mixup(
    images: tf.Tensor,
    labels: tf.Tensor,
    mixup_threshold: float = 0.5,
    image_dimension: int = 3,
    beta: float = 1.0,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """implementation of mixup augmentation.
    See [mixup: Beyond Empirical Risk Minimization] (https://arxiv.org/abs/1710.09412v2)

    Args:
        images (tf.Tensor): Batch of images to process.
        labels (tf.Tensor): Labels of images to process. This function excepted labels to be one-hot encoded.
        mixup_threshold (float, optional): Defaults to 0.5.
        image_dimension (int, optional): Defaults to 3.
        beta (float, optional): Positive (>0), Defaults to 1.0.

    Raises:
        ValueError: if dimesion of images is less than image_dimension+1.

    Returns:
        Tuple[tf.Tensor,tf.Tensor]: Tuple[images, labels]
    """

    batch_size = tf.shape(images)[0]

    if images.shape.ndims != image_dimension + 1:
        raise ValueError(f"Only batch of images is accepted. Input Images Dimesnion must be {image_dimension+1}")

    pairs = _make_pairs(batch_size, mixup_threshold)

    if beta > 0:
        combination_ratio = _estimated_beta_distribution(beta, beta, size=batch_size)
    else:
        combination_ratio = tf.ones([batch_size])

    original_images_type = images.dtype
    if images.dtype != tf.float32:
        images = tf.image.convert_image_dtype(images, tf.float32)
    if labels.dtype != tf.float32:
        labels = tf.cast(labels, tf.float32)

    # pylint: disable=no-value-for-parameter
    images = tf.einsum("b...,b->b...", images, combination_ratio) + tf.einsum(
        "b...,b->b...", tf.gather(images, pairs[:, 1]), 1 - combination_ratio
    )
    # pylint: disable=no-value-for-parameter
    labels = tf.einsum("b...,b->b...", labels, combination_ratio) + tf.einsum(
        "b...,b->b...", tf.gather(labels, pairs[:, 1]), 1 - combination_ratio
    )

    if images.dtype != original_images_type:
        images = tf.image.convert_image_dtype(images, original_images_type)

    return images, labels


def random_erasing(
    image: tf.Tensor,
    label: tf.Tensor,
    random_erasing_threshold: float = 0.5,
    scale: Tuple[float, float] = (0.02, 0.4),
    ratio: Tuple[float, float] = (0.3, 3.3),
    value: Union[Tuple[float, float, float], float, str] = (0.4914, 0.4822, 0.4465),
    max_attempt: int = 50,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """implementattion of random erasing augmentation.
    See [Random Erasing Data Augmentation] (https://arxiv.org/abs/1708.04896)

    Args:
        image (tf.Tensor): image to process
        label (tf.Tensor): label of the image
        random_erasing_threshold (float, optional): Defaults to 0.5.
        scale (Tuple[float, float], optional): Defaults to (0.02, 0.4).
        ratio (Tuple[float, float], optional): Defaults to (0.3, 3.3).
        value (Union[Tuple[float, float, float], float, str], optional): Defaults to (0.4914, 0.4822, 0.4465).
        max_attempt (int, optional): Defaults to 50.

    Raises:
        ValueError: if value is string and not in ["random","black","white"].

    Returns:
        Tuple[tf.Tensor,tf.Tensor]: Tuple[image, label]
    """
    accepted_str_values = ("random", "black", "white")

    shape = image.shape
    height = shape[0]
    width = shape[1]

    def _erase(image, y1, x1, h, w, value):
        crop_blank = tf.image.pad_to_bounding_box(
            tf.image.crop_to_bounding_box(image, y1, x1, h, w), y1, x1, height, width
        )
        image = image - crop_blank
        value_add = tf.image.pad_to_bounding_box(value, y1, x1, height, width)
        return image + value_add

    if isinstance(value, str):
        if value not in accepted_str_values:
            raise ValueError("if value is str, it should be random or black or white")

    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    if tf.random.uniform([]) <= random_erasing_threshold:
        return image, label

    for _ in tf.range(max_attempt):
        area = height * width
        # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        target_area = tf.random.uniform([], minval=scale[0], maxval=scale[1]) * area
        aspect_ratio = tf.random.uniform([], minval=ratio[0], maxval=ratio[1])

        h = tf.cast(tf.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32)
        w = tf.cast(tf.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32)

        if w < width and h < height:
            x1 = tf.random.uniform([], minval=0, maxval=width - w, dtype=tf.int32)
            y1 = tf.random.uniform([], minval=0, maxval=height - h, dtype=tf.int32)

            if value == "black":
                value_add = tf.image.convert_image_dtype(tf.cast(tf.fill([h, w, shape[2]], 0), tf.uint8), image.dtype)
                image = _erase(image, y1, x1, h, w, value_add)

            if value == "white":
                value_add = tf.image.convert_image_dtype(
                    tf.cast(tf.fill([h, w, shape[2]], 255), tf.uint8), image.dtype
                )
                image = _erase(image, y1, x1, h, w, value_add)

            if value == "random":
                value_add = tf.image.convert_image_dtype(tf.random.normal([h, w, shape[2]]), image.dtype)
                image = _erase(image, y1, x1, h, w, value_add)

            if isinstance(value, tuple):
                r = tf.fill([h, w], value[0])
                g = tf.fill([h, w], value[1])
                b = tf.fill([h, w], value[2])
                value_add = tf.image.convert_image_dtype(tf.stack([r, g, b], axis=-1), image.dtype)
                image = _erase(image, y1, x1, h, w, value_add)

            if isinstance(value, float):
                value_add = tf.image.convert_image_dtype(tf.fill([h, w, shape[2]], value), image.dtype)
                image = _erase(image, y1, x1, h, w, value_add)
            break
    return image, label
