from __future__ import annotations

import numpy as np
import pytest

from src.synthesize.poisson_edit.poisson_blend import (
    copy_paste_3d,
    match_source_intensity,
    poisson_blend_3d,
    seam_metric,
)


def _synthetic_inputs():
    z, y, x = np.indices((12, 12, 12))
    source = 10.0 + 2.0 * z + 0.5 * y - 0.25 * x
    mask = np.zeros_like(source, dtype=bool)
    mask[4:8, 4:8, 4:8] = True
    target = np.full((24, 24, 24), 100.0, dtype=np.float64)
    return source, target, mask, (6, 6, 6)


def test_poisson_blend_is_finite_converged_and_preserves_exterior():
    source, target, mask, offset = _synthetic_inputs()
    blended, diagnostics = poisson_blend_3d(source, target, mask, offset)

    inserted = np.zeros_like(target, dtype=bool)
    roi = tuple(slice(start, start + size) for start, size in zip(offset, mask.shape))
    inserted[roi] = mask
    assert np.array_equal(blended[~inserted], target[~inserted])
    assert np.all(np.isfinite(blended))
    assert diagnostics["unknown_voxels"] == int(mask.sum())
    assert diagnostics["relative_residual"] <= 1e-5


def test_poisson_reduces_constant_offset_seam_against_copy_paste():
    source, target, mask, offset = _synthetic_inputs()
    direct = copy_paste_3d(source, target, mask, offset)
    blended, _ = poisson_blend_3d(source, target, mask, offset)

    assert seam_metric(blended, mask, offset) < seam_metric(direct, mask, offset)


def test_intensity_matching_uses_ring_statistics():
    source, _, mask, _ = _synthetic_inputs()
    target_patch = source * 3.0 + 50.0
    matched, stats = match_source_intensity(source, target_patch, mask)

    assert np.allclose(matched, target_patch)
    assert stats["intensity_scale"] == pytest.approx(3.0)


def test_empty_or_edge_touching_mask_is_rejected():
    source, target, mask, offset = _synthetic_inputs()
    with pytest.raises(ValueError, match="empty"):
        poisson_blend_3d(source, target, np.zeros_like(mask), offset)

    mask[0, 4, 4] = True
    with pytest.raises(ValueError, match="boundary"):
        poisson_blend_3d(source, target, mask, offset)
