"""FastSurfer 皮层约束的病灶放置。

病灶中心先验：从 FastSurfer 分割中标记为皮层（ctx-lh-/ctx-rh-）的体素里
预计算合法的病灶中心集合，每次尝试从中均匀抽取。不要求病灶整体落在皮层内
（脑膜瘤可自皮层向外延伸），不引入曲面、皮层厚度或区域先验。
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


def compute_center_candidates(cortical_mask, max_patch_shape) -> np.ndarray:
    """每例预计算合法的病灶中心集合（中心体素必须属于皮层）。

    只做与供体无关的先验过滤：
    1. 中心体素属于皮层（直接从皮层体素中筛选）；
    2. 以最大病灶补丁尺寸 max_patch_shape 居中放置时补丁不越界。
    实际供体的补丁尺寸不超过 max_patch_shape（transform_donor_mask 保持
    形状不变），其 ROI 是最大补丁 ROI 的子集，因此自动满足越界约束；
    与已有标签保护区的重叠在 choose_cortical_candidate 中按当前标签校验。
    """
    cortical = np.asarray(cortical_mask, dtype=bool)
    max_patch = np.asarray(max_patch_shape, dtype=np.int64)
    if max_patch.shape != (3,) or np.any(max_patch <= 0):
        raise ValueError(
            f"max patch shape must contain three positive values: {max_patch_shape}"
        )
    half = max_patch // 2
    image_shape = np.asarray(cortical.shape)
    centers = np.argwhere(cortical)
    if len(centers) == 0:
        raise RuntimeError("cortical mask contains no candidate voxels")
    low_ok = centers >= half
    high_ok = centers <= image_shape - (max_patch - half)
    centers = centers[(low_ok & high_ok).all(axis=1)]
    if len(centers) == 0:
        raise RuntimeError(
            "no cortical center can fit the maximum patch shape "
            f"{tuple(int(v) for v in max_patch)} inside the volume"
        )
    return centers


def choose_cortical_candidate(
    center_candidates,
    existing_label,
    donor_mask,
    rng,
    *,
    protected_dilation: int,
    max_attempts: int,
):
    """从预计算的合法中心集合中均匀抽取候选，返回 (center, roi)。

    中心在皮层内由候选集合的构造保证（见 compute_center_candidates）；
    越界约束由最大补丁尺寸的预过滤保证，此处仅做防御性复查，并校验病灶
    补丁是否与已有标签（含先前插入的合成病灶）的保护区域重叠。
    """
    existing = np.asarray(existing_label)
    donor = np.asarray(donor_mask, dtype=bool)
    if not donor.any():
        raise ValueError("donor mask is empty")
    protected = binary_dilation(existing > 0, iterations=int(protected_dilation))
    candidates = np.asarray(center_candidates, dtype=np.int64)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or len(candidates) == 0:
        raise ValueError("center candidates must be a non-empty Nx3 array")
    patch_shape = np.asarray(donor.shape)
    half = patch_shape // 2
    image_shape = np.asarray(existing.shape)

    protected_overlap = 0
    for _ in range(int(max_attempts)):
        center = candidates[int(rng.integers(0, len(candidates)))].copy()
        start = center - half
        end = start + patch_shape
        if np.any(start < 0) or np.any(end > image_shape):
            continue  # 防御性检查：实际供体补丁不应超过预计算的最大尺寸
        roi = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
        if np.any(protected[roi][donor]):
            protected_overlap += 1
            continue
        return tuple(int(value) for value in center), roi
    raise RuntimeError(
        "no cortical lesion placement found after "
        f"{max_attempts} attempts: protected_overlap={protected_overlap}"
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
