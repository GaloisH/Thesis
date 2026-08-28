"""Lesion placement, sampling, quality control, and synthesis pipeline."""

from .pipeline import synthesize, synthesize_legacy, synthesize_target
from .cortical_placement import (
    choose_cortical_candidate,
    compute_center_candidates,
    load_cortical_mask,
    placement_report,
    read_cortical_label_ids,
)
from .placement import transform_donor_mask
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
    "compute_center_candidates",
    "hard_composite",
    "load_cortical_mask",
    "placement_report",
    "qc_patch",
    "read_cortical_label_ids",
    "sample_composite_patch",
    "sample_histogram",
    "synthesize",
    "synthesize_legacy",
    "synthesize_target",
    "transform_donor_mask",
]
