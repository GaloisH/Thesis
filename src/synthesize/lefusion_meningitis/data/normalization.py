from __future__ import annotations

import math
from typing import Any

import numpy as np


def robust_normalize(image, clip_z: float = 5.0, epsilon: float = 1e-6):
    """Z-score finite brain foreground and scale the clipped values to [-1, 1]."""
    image = np.asarray(image, dtype=np.float32)
    foreground = np.isfinite(image) & (np.abs(image) > epsilon)
    if np.count_nonzero(foreground) < 32:
        foreground = np.isfinite(image)
    values = image[foreground]
    if values.size == 0:
        raise ValueError("image contains no finite foreground")
    mean = float(values.mean())
    std = float(values.std())
    if not math.isfinite(std) or std <= epsilon:
        raise ValueError("image foreground has near-zero standard deviation")
    normalized = np.clip((image - mean) / std, -clip_z, clip_z) / clip_z
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32), {
        "mean": mean,
        "std": std,
        "clip_z": float(clip_z),
        "foreground_voxels": int(values.size),
    }


def denormalize_image(image, metadata: dict[str, Any]):
    """Restore image intensities using saved normalization metadata."""
    return (
        np.asarray(image, dtype=np.float32)
        * float(metadata["clip_z"])
        * float(metadata["std"])
        + float(metadata["mean"])
    )


def lesion_histogram(image, mask, bins: int = 16):
    """Compute a fixed-length histogram over normalized lesion voxels."""
    values = np.asarray(image)[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        raise ValueError("cannot compute a histogram for an empty lesion")
    histogram, _ = np.histogram(values, bins=bins, range=(-1.0, 1.0))
    histogram = histogram.astype(np.float32)
    histogram /= max(float(histogram.sum()), 1.0)
    return histogram
