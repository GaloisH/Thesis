"""Lesion placement, sampling, quality control, and synthesis pipeline."""

from .pipeline import synthesize
from .cortical_placement import (
    choose_cortical_candidate,
    load_cortical_mask,
    placement_report,
    read_cortical_label_ids,
)
from .placement import roi_from_mask, transform_donor_mask
from .quality import qc_patch
from .sampling import (
    brighten_lesion_interior,
    hard_composite,
    sample_composite_patch,
    sample_histogram,
)

__all__ = [
    "brighten_lesion_interior",
    "choose_cortical_candidate",
    "hard_composite",
    "load_cortical_mask",
    "placement_report",
    "qc_patch",
    "read_cortical_label_ids",
    "roi_from_mask",
    "sample_composite_patch",
    "sample_histogram",
    "synthesize",
    "transform_donor_mask",
]
