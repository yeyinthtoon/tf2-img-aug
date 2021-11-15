"""This module contains useful augmentation methods for classification."""
from .augment import mixup, cutmix, random_erasing

__all__ = ["mixup", "cutmix", "random_erasing"]
