from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
from scipy.ndimage import label

from tqdm import tqdm

from ..io import label_path, load_ras
from ..logger import get_logger

logger = get_logger(__name__)


def centered_crop_with_padding(
    array,
    center,
    shape,
    *,
    pad_mode: str,
    pad_value: float = 0,
):
    """裁剪数组到指定形状，必要时在边界上进行填充。"""
    array = np.asarray(array)
    shape = np.asarray(shape, dtype=np.int64)
    center = np.asarray(center, dtype=np.int64)
    start = center - shape // 2
    end = start + shape
    before = np.maximum(-start, 0)
    after = np.maximum(end - np.asarray(array.shape), 0)
    source_start = np.maximum(start, 0)
    source_end = np.minimum(end, np.asarray(array.shape))
    slices = tuple(slice(int(a), int(b)) for a, b in zip(source_start, source_end))
    cropped = array[slices]
    padding = tuple((int(a), int(b)) for a, b in zip(before, after))
    kwargs = {"constant_values": pad_value} if pad_mode == "constant" else {}
    cropped = np.pad(cropped, padding, mode=pad_mode, **kwargs)
    if tuple(cropped.shape) != tuple(shape):
        raise AssertionError(f"crop shape {cropped.shape} != requested {tuple(shape)}")
    return cropped, {
        "start": [int(value) for value in start],
        "end": [int(value) for value in end],
        "padding": [[int(a), int(b)] for a, b in padding],
    }


def component_records(mask, min_voxels: int = 1):
    """提取二进制掩码的连通组件信息。"""
    components, count = label(np.asarray(mask, dtype=bool))
    records: list[dict[str, Any]] = []
    for component_id in range(1, count + 1):
        coordinates = np.argwhere(components == component_id)
        if len(coordinates) < min_voxels:
            continue
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0) + 1
        records.append(
            {
                "component_id": int(component_id),
                "voxels": int(len(coordinates)),
                "centroid": coordinates.mean(axis=0).tolist(),
                "bbox_min": minimum.tolist(),
                "bbox_max": maximum.tolist(),
                "bbox_shape": (maximum - minimum).tolist(),
            }
        )
    return components, records


def collect_case_statistics(
    dataset_dir: str | Path,
    cases: list[str],
    channel: int,
    label_id: int,
):
    """Collect lesion voxel and connected-component counts for each case."""
    logger.info(
        "Collecting lesion statistics for %d cases (channel=%d, label=%d)",
        len(cases),
        channel,
        label_id,
    )
    statistics: list[dict[str, Any]] = []
    for case_id in tqdm(cases, desc="Collecting case statistics", unit="case"):
        label_array, _ = load_ras(label_path(dataset_dir, case_id), label=True)
        foreground = label_array == label_id
        _, records = component_records(foreground)
        statistics.append(
            {
                "case_id": case_id,
                "lesion_voxels": int(foreground.sum()),
                "components": int(len(records)),
            }
        )
    logger.info(
        "Statistics collected: %d total lesion voxels, %d total components",
        sum(item["lesion_voxels"] for item in statistics),
        sum(item["components"] for item in statistics),
    )
    return statistics
