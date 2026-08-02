"""Fixed-mask synthesis visualization and publication-ready reporting."""

from .command import visualize
from .plots import masked_absolute_difference

__all__ = ["masked_absolute_difference", "visualize"]
