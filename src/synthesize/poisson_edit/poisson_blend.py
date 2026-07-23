"""Sparse three-dimensional Poisson image editing.

The implementation solves the classic seamless-cloning objective on an
irregular 3-D lesion domain.  Only voxels inside ``lesion_mask`` are unknown;
the target volume supplies Dirichlet boundary values.
"""

from __future__ import annotations

from itertools import product
from typing import Any
import warnings

import numpy as np
from scipy import sparse
from scipy.ndimage import binary_dilation
from scipy.sparse.linalg import MatrixRankWarning, spsolve


NEIGHBORS_3D = tuple(
    delta
    for delta in product((-1, 0, 1), repeat=3)
    if sum(abs(value) for value in delta) == 1
)


def _validate_inputs(
    source_patch: np.ndarray,
    target_volume: np.ndarray,
    lesion_mask: np.ndarray,
    offset: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[slice, slice, slice]]:
    source = np.asarray(source_patch, dtype=np.float64)
    target = np.asarray(target_volume)
    mask = np.asarray(lesion_mask, dtype=bool)

    if source.ndim != 3 or target.ndim != 3 or mask.ndim != 3:
        raise ValueError("source_patch, target_volume and lesion_mask must be 3-D")
    if source.shape != mask.shape:
        raise ValueError("source_patch and lesion_mask must have identical shapes")
    if not mask.any():
        raise ValueError("lesion_mask is empty")
    if len(offset) != 3:
        raise ValueError("offset must contain three z-y-x coordinates")

    for axis in range(3):
        edge_selector = [slice(None)] * 3
        edge_selector[axis] = 0
        if mask[tuple(edge_selector)].any():
            raise ValueError("lesion_mask touches a patch boundary")
        edge_selector[axis] = -1
        if mask[tuple(edge_selector)].any():
            raise ValueError("lesion_mask touches a patch boundary")

    starts = tuple(int(value) for value in offset)
    stops = tuple(start + size for start, size in zip(starts, source.shape))
    if any(start < 0 for start in starts) or any(
        stop > size for stop, size in zip(stops, target.shape)
    ):
        raise ValueError("source patch placement lies outside target_volume")

    roi = tuple(slice(start, stop) for start, stop in zip(starts, stops))
    return source, target, mask, roi


def _iqr(values: np.ndarray) -> float:
    q25, q75 = np.percentile(values, (25.0, 75.0))
    return float(q75 - q25)


def match_source_intensity(
    source_patch: np.ndarray,
    target_patch: np.ndarray,
    lesion_mask: np.ndarray,
    *,
    ring_inner: int = 1,
    ring_outer: int = 5,
    eps: float = 1e-6,
) -> tuple[np.ndarray, dict[str, float]]:
    """Match donor and target context using robust median/IQR statistics."""
    source = np.asarray(source_patch, dtype=np.float64)
    target = np.asarray(target_patch, dtype=np.float64)
    mask = np.asarray(lesion_mask, dtype=bool)
    if source.shape != target.shape or source.shape != mask.shape:
        raise ValueError("source, target and mask shapes must match")
    if not mask.any():
        raise ValueError("lesion_mask is empty")
    if ring_inner < 0 or ring_outer <= ring_inner:
        raise ValueError("ring_outer must be greater than ring_inner")

    inner = binary_dilation(mask, iterations=ring_inner) if ring_inner else mask
    outer = binary_dilation(mask, iterations=ring_outer)
    ring = outer & ~inner
    if np.count_nonzero(ring) < 16:
        raise ValueError("insufficient exterior ring voxels for intensity matching")

    source_values = source[ring]
    target_values = target[ring]
    source_median = float(np.median(source_values))
    target_median = float(np.median(target_values))
    source_iqr = _iqr(source_values)
    target_iqr = _iqr(target_values)
    if source_iqr <= eps or target_iqr <= eps:
        raise ValueError("source or target exterior ring has near-zero IQR")

    scale = target_iqr / source_iqr
    matched = (source - source_median) * scale + target_median
    stats = {
        "source_ring_median": source_median,
        "source_ring_iqr": source_iqr,
        "target_ring_median": target_median,
        "target_ring_iqr": target_iqr,
        "intensity_scale": float(scale),
    }
    return matched, stats


def copy_paste_3d(
    source_patch: np.ndarray,
    target_volume: np.ndarray,
    lesion_mask: np.ndarray,
    offset: tuple[int, int, int],
) -> np.ndarray:
    """Copy source lesion voxels into a target volume without blending."""
    source, target, mask, roi = _validate_inputs(
        source_patch, target_volume, lesion_mask, offset
    )
    result = np.asarray(target, dtype=np.float64).copy()
    target_roi = result[roi]
    target_roi[mask] = source[mask]
    return result


def poisson_blend_3d(
    source_patch: np.ndarray,
    target_volume: np.ndarray,
    lesion_mask: np.ndarray,
    offset: tuple[int, int, int],
    *,
    residual_tolerance: float = 1e-5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Blend a lesion by solving a masked 3-D Poisson system with SciPy."""
    source, target, mask, roi = _validate_inputs(
        source_patch, target_volume, lesion_mask, offset
    )
    target_float = np.asarray(target, dtype=np.float64)
    target_patch = target_float[roi]

    coordinates = np.argwhere(mask)
    unknown_count = int(coordinates.shape[0])
    index_map = np.full(mask.shape, -1, dtype=np.int32)
    index_map[tuple(coordinates.T)] = np.arange(unknown_count, dtype=np.int32)

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    rhs = np.zeros(unknown_count, dtype=np.float64)

    for row, coordinate_array in enumerate(coordinates):
        coordinate = tuple(int(value) for value in coordinate_array)
        rows.append(row)
        cols.append(row)
        values.append(6.0)
        source_center = source[coordinate]

        for delta in NEIGHBORS_3D:
            neighbor = tuple(
                coordinate[axis] + delta[axis] for axis in range(3)
            )
            rhs[row] += source_center - source[neighbor]
            neighbor_index = int(index_map[neighbor])
            if neighbor_index >= 0:
                rows.append(row)
                cols.append(neighbor_index)
                values.append(-1.0)
            else:
                rhs[row] += target_patch[neighbor]

    matrix = sparse.coo_matrix(
        (values, (rows, cols)), shape=(unknown_count, unknown_count)
    ).tocsr()
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        solution = spsolve(matrix, rhs)

    if not np.all(np.isfinite(solution)):
        raise RuntimeError("Poisson solver produced non-finite values")
    residual = float(
        np.linalg.norm(matrix @ solution - rhs) / max(np.linalg.norm(rhs), 1.0)
    )
    if residual > residual_tolerance:
        raise RuntimeError(
            f"Poisson relative residual {residual:.3e} exceeds "
            f"tolerance {residual_tolerance:.3e}"
        )

    result = target_float.copy()
    result_roi = result[roi]
    result_roi[mask] = solution
    diagnostics: dict[str, Any] = {
        "unknown_voxels": unknown_count,
        "matrix_nonzeros": int(matrix.nnz),
        "relative_residual": residual,
    }
    return result, diagnostics


def seam_metric(
    volume: np.ndarray,
    lesion_mask: np.ndarray,
    offset: tuple[int, int, int],
) -> float:
    """Mean absolute intensity jump across the inserted mask boundary."""
    data = np.asarray(volume, dtype=np.float64)
    mask = np.asarray(lesion_mask, dtype=bool)
    starts = tuple(int(value) for value in offset)
    jumps: list[np.ndarray] = []

    for axis in range(3):
        first = [slice(None)] * 3
        second = [slice(None)] * 3
        first[axis] = slice(0, -1)
        second[axis] = slice(1, None)
        first_t = tuple(first)
        second_t = tuple(second)
        boundary = mask[first_t] ^ mask[second_t]
        if not boundary.any():
            continue

        roi = tuple(
            slice(start, start + size) for start, size in zip(starts, mask.shape)
        )
        patch = data[roi]
        difference = np.abs(patch[first_t] - patch[second_t])
        jumps.append(difference[boundary])

    if not jumps:
        return 0.0
    return float(np.mean(np.concatenate(jumps)))
