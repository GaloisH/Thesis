from __future__ import annotations

from scipy.ndimage import rotate, zoom
from scipy.ndimage import binary_dilation, binary_fill_holes, label
import numpy as np


def transform_donor_mask(mask, rng, rotation_deg: float, scale_range):
    """Apply a light random 3D rotation and isotropic scaling to a donor mask."""
    transformed = np.asarray(mask, dtype=np.float32)
    axes = ((0, 1), (0, 2), (1, 2))[int(rng.integers(0, 3))]
    transformed = rotate(
        transformed,
        float(rng.uniform(-rotation_deg, rotation_deg)),
        axes=axes,
        reshape=False,
        order=0,
        mode="constant",
    )
    scale = float(rng.uniform(*scale_range))
    if abs(scale - 1.0) > 1e-6:
        scaled = zoom(transformed, scale, order=0)
        result = np.zeros_like(transformed)
        source_shape = np.asarray(scaled.shape)
        target_shape = np.asarray(result.shape)
        source_start = np.maximum((source_shape - target_shape) // 2, 0)
        target_start = np.maximum((target_shape - source_shape) // 2, 0)
        extent = np.minimum(source_shape, target_shape)
        source = tuple(
            slice(int(start), int(start + size))
            for start, size in zip(source_start, extent)
        )
        target = tuple(
            slice(int(start), int(start + size))
            for start, size in zip(target_start, extent)
        )
        result[target] = scaled[source]
        transformed = result
    result = transformed > 0.5
    if not result.any():
        raise RuntimeError("donor mask transformation produced an empty mask")
    return result


def choose_candidate(
    image,
    existing_label,
    donor_mask,
    position_centers,
    rng,
    *,
    protected_dilation: int,
    max_attempts: int,
):
    """选择一个解剖上有效的候选位置来放置合成病灶。"""
    finite = np.isfinite(image)
    nonzero = finite & (np.abs(image) > 1e-6)
    components, count = label(nonzero)
    if count:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        brain = binary_fill_holes(components == int(sizes.argmax()))
    else:
        brain = finite
    protected = binary_dilation(existing_label > 0, iterations=protected_dilation)
    patch_shape = np.asarray(donor_mask.shape)
    half = patch_shape // 2
    centers = np.asarray(position_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
        raise ValueError("position prior must contain Nx3 centers")

    for _ in range(max_attempts):
        fraction = centers[int(rng.integers(0, len(centers)))].copy()
        fraction += rng.normal(0.0, 0.02, size=3)
        fraction = np.clip(fraction, 0.0, 1.0)
        center = np.rint(fraction * (np.asarray(image.shape) - 1)).astype(np.int64)
        start = center - half
        end = start + patch_shape
        if np.any(start < 0) or np.any(end > np.asarray(image.shape)):
            continue
        roi = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
        if not np.all(brain[roi][donor_mask]):
            continue
        if np.any(protected[roi][donor_mask]):
            continue
        return tuple(int(value) for value in center), roi
    raise RuntimeError("no anatomically valid lesion placement found")


def roi_from_mask(mask, patch_shape, *, margin: int = 0):
    """Return an unpadded fixed-size ROI centered on a full-volume lesion mask."""
    lesion = np.asarray(mask, dtype=bool)
    if lesion.ndim != 3:
        raise ValueError(f"lesion mask must be 3D, got shape {lesion.shape}")
    coordinates = np.argwhere(lesion)
    if coordinates.size == 0:
        raise ValueError("lesion mask is empty")
    patch = np.asarray(patch_shape, dtype=np.int64)
    if patch.shape != (3,) or np.any(patch <= 0):
        raise ValueError(f"patch shape must contain three positive values: {patch_shape}")
    margin = int(margin)
    if margin < 0 or np.any(patch <= 2 * margin):
        raise ValueError("mask margin is incompatible with patch shape")

    bbox_min = coordinates.min(axis=0)
    bbox_max = coordinates.max(axis=0) + 1
    bbox_shape = bbox_max - bbox_min
    available = patch - 2 * margin
    if np.any(bbox_shape > available):
        raise ValueError(
            "lesion mask does not fit inside the model patch with margin: "
            f"bbox={tuple(int(v) for v in bbox_shape)}, "
            f"available={tuple(int(v) for v in available)}"
        )
    center = np.floor((bbox_min + bbox_max - 1) / 2.0).astype(np.int64)
    start = center - patch // 2
    end = start + patch
    if np.any(start < 0) or np.any(end > np.asarray(lesion.shape)):
        raise ValueError("lesion is too close to the image edge for an unpadded model patch")
    if np.any(bbox_min < start + margin) or np.any(bbox_max > end - margin):
        raise ValueError(
            "lesion cannot be centered in the model patch with the requested margin"
        )
    roi = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
    return roi, {
        "center": [int(v) for v in center],
        "start": [int(v) for v in start],
        "end": [int(v) for v in end],
        "bbox_min": [int(v) for v in bbox_min],
        "bbox_max": [int(v) for v in bbox_max],
        "bbox_shape": [int(v) for v in bbox_shape],
        "margin": margin,
    }
