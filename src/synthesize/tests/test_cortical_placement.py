"""cortical_placement 模块单元测试（计划验证项 1-4）。

使用合成的微型 NumPy/NIfTI 体积，不依赖真实医学数据。
可直接以 `python test_cortical_placement.py` 运行，也可由 pytest 收集。
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nibabel.freesurfer.mghformat import MGHImage

from lefusion_meningitis.synthesis.cortical_placement import (
    choose_cortical_candidate,
    load_cortical_mask,
    placement_report,
    read_cortical_label_ids,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FASTSURFER_LUT = (
    REPO_ROOT
    / "FastSurfer"
    / "FastSurferCNN"
    / "config"
    / "FastSurfer_ColorLUT.tsv"
)


@contextmanager
def _temp_dir():
    directory = Path(tempfile.mkdtemp())
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _save_mgz(path: Path, data, affine) -> None:
    MGHImage(np.asarray(data, dtype=np.float32), np.asarray(affine)).to_filename(
        str(path)
    )


def test_read_cortical_label_ids_repo_lut():
    ids = read_cortical_label_ids(FASTSURFER_LUT)
    assert 1002 in ids  # ctx-lh-caudalanteriorcingulate
    assert 2002 in ids  # ctx-rh-caudalanteriorcingulate
    assert 2 not in ids  # white matter
    assert 0 not in ids  # background
    assert ids.ndim == 1
    assert len(np.unique(ids)) == len(ids)
    assert (np.diff(ids) > 0).all()  # sorted


def test_read_cortical_label_ids_missing_file():
    with _temp_dir() as directory:
        try:
            read_cortical_label_ids(directory / "missing.tsv")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError")


def test_read_cortical_label_ids_bad_header():
    with _temp_dir() as directory:
        path = directory / "bad.tsv"
        path.write_text("A\tB\n1\tx\n", encoding="utf-8")
        try:
            read_cortical_label_ids(path)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_read_cortical_label_ids_no_cortex():
    with _temp_dir() as directory:
        path = directory / "no_cortex.tsv"
        path.write_text(
            "ID\tLabelName\tR\tG\tB\tA\n"
            "2\tLeft-Cerebral-White-Matter\t245\t245\t245\t0\n",
            encoding="utf-8",
        )
        try:
            read_cortical_label_ids(path)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_load_cortical_mask_aligned():
    import nibabel as nib

    shape = (24, 24, 24)
    affine = np.eye(4)
    reference = nib.Nifti1Image(np.zeros(shape), affine)
    segmentation = np.zeros(shape, dtype=np.int16)
    segmentation[5:15, 5:15, 5:15] = 1002  # cortical
    segmentation[2:4, 2:4, 2:4] = 2  # white matter
    with _temp_dir() as directory:
        seg_path = directory / "aparc.DKTatlas+aseg.deep.mgz"
        _save_mgz(seg_path, segmentation, affine)
        mask = load_cortical_mask(seg_path, reference, FASTSURFER_LUT)
    assert mask.shape == shape
    assert mask[6, 6, 6]
    assert not mask[3, 3, 3]
    assert not mask[0, 0, 0]


def test_load_cortical_mask_resamples_to_reference_grid():
    import nibabel as nib

    ref_shape = (24, 24, 24)
    ref_affine = np.diag([1.5, 1.5, 1.5, 1.0])
    reference = nib.Nifti1Image(np.zeros(ref_shape), ref_affine)
    coarse_affine = np.diag([3.0, 3.0, 3.0, 1.0])  # 2x voxel size
    coarse = np.zeros((12, 12, 12), dtype=np.int16)
    coarse[3:9, 3:9, 3:9] = 1002
    with _temp_dir() as directory:
        seg_path = directory / "seg.nii.gz"
        nib.Nifti1Image(coarse, coarse_affine).to_filename(str(seg_path))
        mask = load_cortical_mask(seg_path, reference, FASTSURFER_LUT)
    assert mask.shape == ref_shape
    assert mask.any()
    assert mask[12, 12, 12]  # 18mm world position, inside the coarse cortical block


def test_load_cortical_mask_empty_raises():
    import nibabel as nib

    shape = (16, 16, 16)
    reference = nib.Nifti1Image(np.zeros(shape), np.eye(4))
    segmentation = np.zeros(shape, dtype=np.int16)
    segmentation[3:6, 3:6, 3:6] = 2  # only white matter
    with _temp_dir() as directory:
        seg_path = directory / "seg.mgz"
        _save_mgz(seg_path, segmentation, np.eye(4))
        try:
            load_cortical_mask(seg_path, reference, FASTSURFER_LUT)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")


def test_load_cortical_mask_missing_file_raises():
    import nibabel as nib

    reference = nib.Nifti1Image(np.zeros((8, 8, 8)), np.eye(4))
    with _temp_dir() as directory:
        try:
            load_cortical_mask(
                directory / "missing.mgz", reference, FASTSURFER_LUT
            )
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError")


def test_choose_cortical_candidate_all_donor_cortical():
    rng = np.random.default_rng(7)
    cortical = np.zeros((30, 30, 30), dtype=bool)
    cortical[6:24, 6:24, 6:24] = True
    existing = np.zeros((30, 30, 30), dtype=np.int16)
    donor = np.zeros((12, 12, 12), dtype=bool)
    donor[4:8, 4:8, 4:8] = True
    center, roi = choose_cortical_candidate(
        cortical,
        existing,
        donor,
        rng,
        protected_dilation=2,
        max_attempts=500,
    )
    assert np.all(cortical[roi][donor])
    assert len(center) == 3


def test_choose_cortical_candidate_deterministic():
    cortical = np.zeros((30, 30, 30), dtype=bool)
    cortical[5:25, 5:25, 5:25] = True
    existing = np.zeros((30, 30, 30), dtype=np.int16)
    donor = np.zeros((12, 12, 12), dtype=bool)
    donor[4:8, 4:8, 4:8] = True
    first = choose_cortical_candidate(
        cortical,
        existing,
        donor,
        np.random.default_rng(99),
        protected_dilation=2,
        max_attempts=500,
    )
    second = choose_cortical_candidate(
        cortical,
        existing,
        donor,
        np.random.default_rng(99),
        protected_dilation=2,
        max_attempts=500,
    )
    assert first[0] == second[0]
    assert all(
        a.start == b.start and a.stop == b.stop for a, b in zip(first[1], second[1])
    )


def test_choose_cortical_candidate_rejects_non_cortical():
    rng = np.random.default_rng(3)
    shape = (24, 24, 24)
    cortical = np.zeros(shape, dtype=bool)
    cortical[4:20, 4:20, 6:10] = True  # 4-deep slab in z, inset in all axes
    existing = np.zeros(shape, dtype=np.int16)
    donor = np.zeros((8, 8, 8), dtype=bool)
    donor[1:7, 1:7, 1:7] = True  # 6-deep in z; can never fit a 4-deep slab
    try:
        choose_cortical_candidate(
            cortical,
            existing,
            donor,
            rng,
            protected_dilation=0,
            max_attempts=300,
        )
    except RuntimeError as exc:
        counters = dict(re.findall(r"(\w+)=(\d+)", str(exc)))
        assert int(counters["outside_cortex"]) > 0
        assert int(counters["out_of_bounds"]) == 0
    else:
        raise AssertionError("expected RuntimeError")


def test_choose_cortical_candidate_rejects_protected_overlap():
    rng = np.random.default_rng(5)
    shape = (24, 24, 24)
    cortical = np.zeros(shape, dtype=bool)
    cortical[4:20, 4:20, 4:20] = True  # frame of 4 keeps centers in-bounds
    existing = np.ones(shape, dtype=np.int16)  # everything is protected
    donor = np.zeros((8, 8, 8), dtype=bool)
    donor[1:7, 1:7, 1:7] = True
    try:
        choose_cortical_candidate(
            cortical,
            existing,
            donor,
            rng,
            protected_dilation=2,
            max_attempts=100,
        )
    except RuntimeError as exc:
        counters = dict(re.findall(r"(\w+)=(\d+)", str(exc)))
        assert int(counters["protected_overlap"]) > 0
        assert int(counters["out_of_bounds"]) == 0
    else:
        raise AssertionError("expected RuntimeError")


def test_choose_cortical_candidate_empty_cortex_raises():
    rng = np.random.default_rng(1)
    cortical = np.zeros((16, 16, 16), dtype=bool)
    existing = np.zeros((16, 16, 16), dtype=np.int16)
    donor = np.zeros((4, 4, 4), dtype=bool)
    donor[1:3, 1:3, 1:3] = True
    try:
        choose_cortical_candidate(
            cortical,
            existing,
            donor,
            rng,
            protected_dilation=0,
            max_attempts=10,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")


def test_placement_report_cortical_fraction_one():
    cortical = np.zeros((30, 30, 30), dtype=bool)
    cortical[8:22, 8:22, 8:22] = True
    donor = np.zeros((10, 10, 10), dtype=bool)
    donor[2:8, 2:8, 2:8] = True
    roi = tuple(slice(10, 20) for _ in range(3))  # inside cortex
    report = placement_report(cortical, roi, donor)
    assert report["cortical_fraction"] == 1.0
    assert report["lesion_voxels"] == int(donor.sum())
    assert report["cortical_voxels"] == int(donor.sum())
    assert report["cortical_candidate_voxels"] == int(cortical.sum())
    assert report["center"] == [15, 15, 15]


def test_placement_report_empty_donor_raises():
    cortical = np.zeros((10, 10, 10), dtype=bool)
    roi = tuple(slice(0, 5) for _ in range(3))
    donor = np.zeros((5, 5, 5), dtype=bool)
    try:
        placement_report(cortical, roi, donor)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


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
