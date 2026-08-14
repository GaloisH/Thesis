"""FastSurfer 皮层约束的病灶放置（首版最小实现）。

只做体素级约束：从 FastSurfer 分割中标记为皮层（ctx-lh-/ctx-rh-）的体素里
选取病灶中心。不引入曲面、皮层厚度或区域先验。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import binary_dilation


def read_cortical_label_ids(lut_path: Path) -> np.ndarray:
    """读取 FastSurfer TSV LUT，返回所有名称以 ctx-lh-/ctx-rh- 开头的 ID。"""
    path = Path(lut_path)
    if not path.is_file():
        raise FileNotFoundError(f"FastSurfer LUT not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    if not lines:
        raise ValueError(f"FastSurfer LUT is empty: {path}")
    header = [column.strip() for column in lines[0].split("\t")]
    if "ID" not in header or "LabelName" not in header:
        raise ValueError(
            f"FastSurfer LUT is missing ID/LabelName columns: {path}"
        )
    id_index = header.index("ID")
    name_index = header.index("LabelName")
    ids: list[int] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) <= max(id_index, name_index):
            continue
        raw_id = columns[id_index].strip()
        name = columns[name_index].strip()
        if raw_id and name.startswith(("ctx-lh-", "ctx-rh-")):
            ids.append(int(raw_id))
    if not ids:
        raise ValueError(f"no ctx-* cortical labels found in LUT: {path}")
    return np.unique(np.asarray(ids, dtype=np.int64))


def load_cortical_mask(
    segmentation_path: Path, image_reference, lut_path: Path
) -> np.ndarray:
    """读取 FastSurfer 分割并返回与参考影像同 shape 的布尔皮层掩膜。"""
    seg_path = Path(segmentation_path)
    if not seg_path.is_file():
        raise FileNotFoundError(f"FastSurfer segmentation not found: {seg_path}")
    segmentation = nib.load(str(seg_path))
    reference = image_reference
    if segmentation.shape == reference.shape and np.allclose(
        segmentation.affine, reference.affine
    ):
        labels = np.asanyarray(segmentation.dataobj)
    else:
        labels = resample_from_to(segmentation, reference, order=0).get_fdata()
    labels = np.asarray(labels, dtype=np.int64)
    cortical_ids = read_cortical_label_ids(lut_path)
    mask = np.isin(labels, cortical_ids)
    if not mask.any():
        raise RuntimeError(
            f"cortical mask is empty for {seg_path} (LUT {lut_path}); "
            "check that the segmentation overlaps the reference image grid"
        )
    return mask


def choose_cortical_candidate(
    cortical_mask,
    existing_label,
    donor_mask,
    rng,
    *,
    protected_dilation: int,
    max_attempts: int,
):
    """从皮层体素均匀抽取候选中心，返回 (center, roi)。"""
    cortical = np.asarray(cortical_mask, dtype=bool)
    existing = np.asarray(existing_label)
    donor = np.asarray(donor_mask, dtype=bool)
    if cortical.shape != existing.shape:
        raise ValueError(
            f"cortical mask shape {cortical.shape} does not match "
            f"existing label shape {existing.shape}"
        )
    if not donor.any():
        raise ValueError("donor mask is empty")
    protected = binary_dilation(existing > 0, iterations=int(protected_dilation))
    patch_shape = np.asarray(donor.shape)
    half = patch_shape // 2
    candidates = np.argwhere(cortical)
    if len(candidates) == 0:
        raise RuntimeError("cortical mask contains no candidate voxels")

    out_of_bounds = 0
    outside_cortex = 0
    protected_overlap = 0
    for _ in range(int(max_attempts)):
        center = candidates[int(rng.integers(0, len(candidates)))].copy()
        start = center - half
        end = start + patch_shape
        if np.any(start < 0) or np.any(end > np.asarray(cortical.shape)):
            out_of_bounds += 1
            continue
        roi = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
        if not np.all(cortical[roi][donor]):
            outside_cortex += 1
            continue
        if np.any(protected[roi][donor]):
            protected_overlap += 1
            continue
        return tuple(int(value) for value in center), roi
    raise RuntimeError(
        "no cortical lesion placement found after "
        f"{max_attempts} attempts: out_of_bounds={out_of_bounds}, "
        f"outside_cortex={outside_cortex}, protected_overlap={protected_overlap}"
    )


def placement_report(cortical_mask, roi, donor_mask) -> dict[str, Any]:
    """返回可写入 metadata 的最小皮层放置验证信息。"""
    cortical = np.asarray(cortical_mask, dtype=bool)
    donor = np.asarray(donor_mask, dtype=bool)
    lesion_voxels = int(donor.sum())
    if lesion_voxels == 0:
        raise ValueError("donor mask is empty")
    cortical_voxels = int(np.count_nonzero(cortical[roi][donor]))
    return {
        "center": [int(s.start + (s.stop - s.start) // 2) for s in roi],
        "cortical_candidate_voxels": int(np.count_nonzero(cortical)),
        "lesion_voxels": lesion_voxels,
        "cortical_voxels": cortical_voxels,
        "cortical_fraction": cortical_voxels / lesion_voxels,
    }
