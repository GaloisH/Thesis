"""Lesion placement, sampling, quality control, and synthesis pipeline."""

from .pipeline import synthesize
from .placement import choose_candidate, roi_from_mask, transform_donor_mask
from .quality import qc_patch
from .sampling import (
    brighten_lesion_interior,
    hard_composite,
    sample_composite_patch,
    sample_histogram,
)

__all__ = [
    "brighten_lesion_interior",
    "choose_candidate",
    "hard_composite",
    "qc_patch",
    "roi_from_mask",
    "sample_composite_patch",
    "sample_histogram",
    "synthesize",
    "transform_donor_mask",
]
