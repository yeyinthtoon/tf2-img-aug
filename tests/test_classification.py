"""Test cases for classification module"""
import tensorflow as tf
from tensorflow import test

from imgaug.classification import cutmix, mixup, random_erasing

delattr(test.TestCase, "test_session")


class TestCutmix(test.TestCase):
    """
    Test cases for cutmix of classification module.
    """

    def test_raise_value_error_if_bad_dims_with_default_argument(self):
        with self.assertRaises(ValueError):
            # pylint: disable=no-value-for-parameter
            images_dummy = tf.random.uniform([220, 220, 3])
            labels = tf.convert_to_tensor([0])
            cutmix(images_dummy, labels)

    def test_raise_value_error_if_bad_dims_with_custom_argument(self):
        with self.assertRaises(ValueError):
            # pylint: disable=no-value-for-parameter
            images_dummy = tf.random.uniform([220, 220])
            labels = tf.convert_to_tensor([0])
            cutmix(images_dummy, labels, image_dimension=2)

    def test_raise_value_error_if_beta_less_than_0(self):
        with self.assertRaises(ValueError):
            # pylint: disable=no-value-for-parameter
            images_dummy = tf.random.uniform([3, 220, 220, 2])
            labels = tf.convert_to_tensor([0, 1, 1])
            cutmix(images_dummy, labels, beta=-0.1)

    def test_raise_value_error_if_beta_is_0(self):
        with self.assertRaises(ValueError):
            # pylint: disable=no-value-for-parameter
            images_dummy = tf.random.uniform([3, 220, 220, 2])
            labels = tf.convert_to_tensor([0, 1, 1])
            cutmix(images_dummy, labels, beta=0.0)

    def test_if_cutmix_threshold_is_one(self):
        # pylint: disable=no-value-for-parameter
        images_dummy = tf.random.uniform([3, 220, 220, 3])
        labels = tf.convert_to_tensor([0, 1, 1])
        images_cutmix, labels = cutmix(images_dummy, labels, cutmix_threshold=1.0)
        self.assertAllClose(images_dummy, images_cutmix)

    def test_if_cutmix_threshold_is_one_autograph(self):
        # pylint: disable=no-value-for-parameter
        images_dummy = tf.random.uniform([3, 220, 220, 3])
        labels = tf.convert_to_tensor([0, 1, 1])
        images_cutmix, labels = tf.function(cutmix)(images_dummy, labels, cutmix_threshold=1.0)
        self.assertAllClose(images_dummy, images_cutmix)


class TestMixup(test.TestCase):
    """
    Test cases for mixup of classification module.
    """

    def test_raise_value_error_if_bad_dims_with_default_argument(self):
        with self.assertRaises(ValueError):
            # pylint: disable=no-value-for-parameter
            images_dummy = tf.random.uniform([220, 220, 3])
            labels = tf.convert_to_tensor([0])
            mixup(images_dummy, labels)

    def test_raise_value_error_if_bad_dims_with_custom_argument(self):
        with self.assertRaises(ValueError):
            # pylint: disable=no-value-for-parameter
            images_dummy = tf.random.uniform([220, 220])
            labels = tf.convert_to_tensor([0])
            mixup(images_dummy, labels, image_dimension=2)

    def test_if_beta_less_than_0(self):
        # pylint: disable=no-value-for-parameter
        images_dummy = tf.random.uniform([3, 220, 220, 3])
        labels = tf.convert_to_tensor([0, 1, 1])
        images_mixup, _ = mixup(images_dummy, labels, beta=-0.1)
        self.assertAllClose(images_dummy, images_mixup)

    def test_if_beta_is_0(self):
        # pylint: disable=no-value-for-parameter
        images_dummy = tf.random.uniform([3, 220, 220, 3])
        labels = tf.convert_to_tensor([0, 1, 1])
        images_mixup, _ = mixup(images_dummy, labels, beta=0.0)
        self.assertAllClose(images_dummy, images_mixup)

    def test_if_mixup_threshold_is_one(self):
        # pylint: disable=no-value-for-parameter
        images_dummy = tf.random.uniform([3, 220, 220, 3])
        labels = tf.convert_to_tensor([0, 1, 1])
        images_mixup, _ = mixup(images_dummy, labels, mixup_threshold=1.0)
        self.assertAllClose(images_dummy, images_mixup)

    def test_return_same_image_type_if_image_is_uint8(self):
        # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
        images_dummy = tf.random.uniform([3, 220, 220, 3], minval=0, maxval=255, dtype=tf.int32)
        images_dummy = tf.cast(images_dummy, tf.uint8)
        labels = tf.convert_to_tensor([0, 1, 1])
        images_mixup, _ = mixup(images_dummy, labels, mixup_threshold=0.5)
        self.assertDTypeEqual(images_mixup, tf.uint8)

    def test_labels_update_if_mixup(self):
        # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
        images_dummy = tf.random.uniform([3, 220, 220, 3], minval=0, maxval=255, dtype=tf.int32)
        images_dummy = tf.cast(images_dummy, tf.uint8)
        labels_dummy = tf.convert_to_tensor([0, 1, 1])
        _, labels = mixup(images_dummy, labels_dummy, mixup_threshold=0.001)
        self.assertNotAllClose(labels_dummy, labels)

    def test_labels_update_if_mixup_autograph(self):
        # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
        images_dummy = tf.random.uniform([3, 220, 220, 3], minval=0, maxval=255, dtype=tf.int32)
        images_dummy = tf.cast(images_dummy, tf.uint8)
        labels_dummy = tf.convert_to_tensor([0, 1, 1])
        _, labels = tf.function(mixup)(images_dummy, labels_dummy, mixup_threshold=0.001)
        self.assertNotAllClose(labels_dummy, labels)

    def test_if_beta_less_than_0_autograph(self):
        # pylint: disable=no-value-for-parameter
        images_dummy = tf.random.uniform([3, 220, 220, 3])
        labels = tf.convert_to_tensor([0, 1, 1])
        images_mixup, _ = tf.function(mixup)(images_dummy, labels, beta=-0.1)
        self.assertAllClose(images_dummy, images_mixup)

    def test_if_beta_is_0_autograph(self):
        # pylint: disable=no-value-for-parameter
        images_dummy = tf.random.uniform([3, 220, 220, 3])
        labels = tf.convert_to_tensor([0, 1, 1])
        images_mixup, _ = tf.function(mixup)(images_dummy, labels, beta=0.0)
        self.assertAllClose(images_dummy, images_mixup)

    def test_if_mixup_threshold_is_one(self):
        # pylint: disable=no-value-for-parameter
        images_dummy = tf.random.uniform([3, 220, 220, 3])
        labels = tf.convert_to_tensor([0, 1, 1])
        images_mixup, _ = tf.function(mixup)(images_dummy, labels, mixup_threshold=1.0)
        self.assertAllClose(images_dummy, images_mixup)

    def test_return_same_image_type_if_image_is_uint8(self):
        # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
        images_dummy = tf.random.uniform([3, 220, 220, 3], minval=0, maxval=255, dtype=tf.int32)
        images_dummy = tf.cast(images_dummy, tf.uint8)
        labels = tf.convert_to_tensor([0, 1, 1])
        images_mixup, _ = tf.function(mixup)(images_dummy, labels, mixup_threshold=0.5)
        self.assertDTypeEqual(images_mixup, tf.uint8)


class TestRandomErasing(test.TestCase):
    """
    Test cases for random erasing of classification module.
    """

    def test_raise_value_error_if_value_is_not_supported(self):
        with self.assertRaises(ValueError):
            # pylint: disable=no-value-for-parameter
            images_dummy = tf.random.uniform([220, 220, 3])
            label = tf.convert_to_tensor([0])
            random_erasing(images_dummy, label, value="test")

    def test_if_random_erasing_threshold_is_one(self):
        # pylint: disable=no-value-for-parameter
        images_dummy = tf.random.uniform([220, 220, 3])
        label = tf.convert_to_tensor([1])
        images_erasing, _ = random_erasing(images_dummy, label, random_erasing_threshold=1.0)
        self.assertAllClose(images_dummy, images_erasing)

    def test_return_same_image_type_if_image_is_uint8(self):
        # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
        images_dummy = tf.random.uniform([220, 220, 3], minval=0, maxval=255, dtype=tf.int32)
        images_dummy = tf.cast(images_dummy, tf.uint8)
        label = tf.convert_to_tensor([1])
        images_erasing, _ = random_erasing(images_dummy, label, random_erasing_threshold=0.5)
        self.assertDTypeEqual(images_erasing, tf.uint8)

    def test_return_same_image_type_if_image_is_float32(self):
        # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
        images_dummy = tf.random.uniform([220, 220, 3], minval=0, maxval=1.0, dtype=tf.float32)
        label = tf.convert_to_tensor([1])
        images_erasing, _ = random_erasing(images_dummy, label, random_erasing_threshold=0.5)
        self.assertDTypeEqual(images_erasing, tf.float32)

    def test_is_augmented_if_value_is_black(self):
        # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
        images_dummy = tf.random.uniform([220, 220, 3], minval=0, maxval=1.0, dtype=tf.float32)
        label = tf.convert_to_tensor([1])
        images_erasing, _ = random_erasing(images_dummy, label, value="black", random_erasing_threshold=0.001)
        self.assertNotAllClose(images_dummy, images_erasing)

    def test_is_augmented_if_value_is_white(self):
        # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
        images_dummy = tf.random.uniform([220, 220, 3], minval=0, maxval=1.0, dtype=tf.float32)
        label = tf.convert_to_tensor([1])
        images_erasing, _ = random_erasing(images_dummy, label, value="white", random_erasing_threshold=0.001)
        self.assertNotAllClose(images_dummy, images_erasing)

    def test_is_augmented_if_value_is_random(self):
        # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
        images_dummy = tf.random.uniform([220, 220, 3], minval=0, maxval=1.0, dtype=tf.float32)
        label = tf.convert_to_tensor([1])
        images_erasing, _ = random_erasing(images_dummy, label, value="random", random_erasing_threshold=0.001)
        self.assertNotAllClose(images_dummy, images_erasing)

    def test_if_random_erasing_threshold_is_one_autograph(self):
        # pylint: disable=no-value-for-parameter
        images_dummy = tf.random.uniform([220, 220, 3])
        label = tf.convert_to_tensor([1])
        images_erasing, _ = tf.function(random_erasing)(images_dummy, label, random_erasing_threshold=1.0)
        self.assertAllClose(images_dummy, images_erasing)

    def test_return_same_image_type_if_image_is_uint8_autograph(self):
        # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
        images_dummy = tf.random.uniform([220, 220, 3], minval=0, maxval=255, dtype=tf.int32)
        images_dummy = tf.cast(images_dummy, tf.uint8)
        label = tf.convert_to_tensor([1])
        images_erasing, _ = tf.function(random_erasing)(images_dummy, label, random_erasing_threshold=0.5)
        self.assertDTypeEqual(images_erasing, tf.uint8)

    def test_return_same_image_type_if_image_is_float32_autograph(self):
        # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
        images_dummy = tf.random.uniform([220, 220, 3], minval=0, maxval=1.0, dtype=tf.float32)
        label = tf.convert_to_tensor([1])
        images_erasing, _ = tf.function(random_erasing)(images_dummy, label, random_erasing_threshold=0.5)
        self.assertDTypeEqual(images_erasing, tf.float32)

    def test_is_augmented_if_value_is_black_autograph(self):
        # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
        images_dummy = tf.random.uniform([220, 220, 3], minval=0, maxval=1.0, dtype=tf.float32)
        label = tf.convert_to_tensor([1])
        images_erasing, _ = tf.function(random_erasing)(
            images_dummy, label, value="black", random_erasing_threshold=0.001
        )
        self.assertNotAllClose(images_dummy, images_erasing)

    def test_is_augmented_if_value_is_white_autograph(self):
        # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
        images_dummy = tf.random.uniform([220, 220, 3], minval=0, maxval=1.0, dtype=tf.float32)
        label = tf.convert_to_tensor([1])
        images_erasing, _ = tf.function(random_erasing)(
            images_dummy, label, value="white", random_erasing_threshold=0.001
        )
        self.assertNotAllClose(images_dummy, images_erasing)

    def test_is_augmented_if_value_is_random_autograph(self):
        # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
        images_dummy = tf.random.uniform([220, 220, 3], minval=0, maxval=1.0, dtype=tf.float32)
        label = tf.convert_to_tensor([1])
        images_erasing, _ = tf.function(random_erasing)(
            images_dummy, label, value="random", random_erasing_threshold=0.001
        )
        self.assertNotAllClose(images_dummy, images_erasing)
