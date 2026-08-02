from __future__ import annotations

from typing import Any
import numpy as np
from scipy.ndimage import binary_dilation

def qc_patch(background, generated, composite, mask, config: dict[str, Any]):
    """检查合成补丁的质量，返回一个字典，指示是否通过 QC 以及失败的原因。"""
    failures: list[str] = []
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 8:
        failures.append("mask_too_small")
    if any(np.any(np.take(mask, (0, -1), axis=axis)) for axis in range(3)):
        failures.append("mask_touches_patch_edge")
    background_exact = bool(np.array_equal(composite[~mask], background[~mask]))
    if not background_exact:
        failures.append("background_changed")
    if not np.all(np.isfinite(composite)):
        failures.append("non_finite")
    if mask.any() and float(np.max(np.abs(generated[mask]))) > float(
        config["intensity_z_limit"]
    ):
        failures.append("generated_intensity_outlier")
    inner = mask & ~binary_dilation(~mask, iterations=1)
    outer = binary_dilation(mask, iterations=1) & ~mask
    boundary_jump = (
        abs(float(composite[inner].mean() - composite[outer].mean()))
        if inner.any() and outer.any()
        else float("inf")
    )
    if boundary_jump > float(config["max_boundary_jump_z"]):
        failures.append("boundary_jump")
    return {
        "passed": not failures,
        "failures": failures,
        "mask_voxels": int(mask.sum()),
        "background_exact": background_exact,
        "boundary_jump": boundary_jump,
    }


def aggregate_case_qc(lesions: list[dict[str, Any]], background_exact: bool):
    """总结一个病例的 QC 结果，返回一个字典，指示是否通过 QC、失败的原因以及其他统计信息。"""
    jumps = [float(item["qc"]["boundary_jump"]) for item in lesions]
    return {
        "passed": bool(lesions) and all(item["qc"]["passed"] for item in lesions),
        "failures": sorted(
            {failure for item in lesions for failure in item["qc"]["failures"]}
        ),
        "background_exact": bool(background_exact),
        "boundary_jump": sum(jumps) / len(jumps) if jumps else None,
        "accepted_lesions": len(lesions),
    }
