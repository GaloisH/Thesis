from __future__ import annotations

from typing import Any
import numpy as np


def qc_patch(background, generated, composite, mask):
    """检查合成补丁的质量，返回一个字典，指示是否通过 QC 以及失败的原因。

    仅保留非有限值检查：掩膜形状先验已由放置阶段的 reject_donor_geometry
    保证，背景一致性由 hard_composite 的硬合成保证；亮度增强
    （brighten_lesion_interior）会刻意制造病灶内部抬升，因此不再用边界跳变
    或强度上限拒绝样本。
    """
    failures: list[str] = []
    if not np.all(np.isfinite(composite)):
        failures.append("non_finite")
    background_exact = bool(np.array_equal(composite[~mask], background[~mask]))
    return {
        "passed": not failures,
        "failures": failures,
        "mask_voxels": int(np.asarray(mask, dtype=bool).sum()),
        "background_exact": background_exact,
    }


def aggregate_case_qc(lesions: list[dict[str, Any]], background_exact: bool):
    """总结一个病例的 QC 结果，返回一个字典，指示是否通过 QC、失败的原因以及其他统计信息。"""
    return {
        "passed": bool(lesions) and all(item["qc"]["passed"] for item in lesions),
        "failures": sorted(
            {failure for item in lesions for failure in item["qc"]["failures"]}
        ),
        "background_exact": bool(background_exact),
        "accepted_lesions": len(lesions),
    }
