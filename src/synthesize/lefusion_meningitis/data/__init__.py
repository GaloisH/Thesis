"""Data preparation, normalization, augmentation, and dataset utilities."""

from .augmentation import augment_pair, build_augmentation
from .components import (
    centered_crop_with_padding,
    collect_case_statistics,
    component_records,
)
from .dataset import MeningitisPatchDataset
from .normalization import denormalize_image, lesion_histogram, robust_normalize
from .preparation import prepare, stratified_split

__all__ = [
    "MeningitisPatchDataset",
    "augment_pair",
    "build_augmentation",
    "centered_crop_with_padding",
    "collect_case_statistics",
    "component_records",
    "denormalize_image",
    "lesion_histogram",
    "prepare",
    "robust_normalize",
    "stratified_split",
]
