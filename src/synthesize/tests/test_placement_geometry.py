"""placement.reject_donor_geometry 单元测试。

使用合成的微型 NumPy 体积，不依赖真实医学数据。
可直接以 `python test_placement_geometry.py` 运行，也可由 pytest 收集。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lefusion_meningitis.synthesis.placement import reject_donor_geometry


def test_reject_donor_geometry_ok():
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[4:8, 4:8, 4:8] = True
    assert reject_donor_geometry(mask, min_voxels=8) is None


def test_reject_donor_geometry_exact_minimum_passes():
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[1:3, 1:3, 1:3] = True  # 8 voxels == threshold, inset from edges
    assert reject_donor_geometry(mask, min_voxels=8) is None


def test_reject_donor_geometry_too_small():
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[:2, :2, :2] = True
    mask[0, 0, 0] = False  # 7 voxels < threshold
    assert reject_donor_geometry(mask, min_voxels=8) == "mask_too_small"


def test_reject_donor_geometry_touches_edge():
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[1:7, 1:7, 1:7] = True
    mask[0, 3, 3] = True  # touches z=0 plane
    assert reject_donor_geometry(mask, min_voxels=8) == "mask_touches_patch_edge"


def test_reject_donor_geometry_empty():
    mask = np.zeros((8, 8, 8), dtype=bool)
    assert reject_donor_geometry(mask, min_voxels=8) == "mask_empty"


def test_reject_donor_geometry_position_independent():
    """平移 mask 不改变结论，因为 ROI 大小恒等于 mask 大小。"""
    base = np.zeros((16, 16, 16), dtype=bool)
    base[4:8, 4:8, 4:8] = True
    shifted = np.zeros((16, 16, 16), dtype=bool)
    shifted[9:13, 9:13, 9:13] = True
    assert reject_donor_geometry(base, min_voxels=8) is None
    assert reject_donor_geometry(shifted, min_voxels=8) is None


def test_reject_donor_geometry_edge_position_independent():
    base = np.zeros((16, 16, 16), dtype=bool)
    base[2:10, 2:10, 2:10] = True
    base[0, 5, 5] = True  # edge touch regardless of translation
    shifted = np.zeros((16, 16, 16), dtype=bool)
    shifted[5:13, 5:13, 5:13] = True
    shifted[0, 8, 8] = True
    assert reject_donor_geometry(base, min_voxels=8) == "mask_touches_patch_edge"
    assert reject_donor_geometry(shifted, min_voxels=8) == "mask_touches_patch_edge"


def _run_all() -> int:
    import traceback

    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}: {exc}")
        else:
            print(f"PASS  {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    for name, exc in failures:
        print(f"--- {name} ---")
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
