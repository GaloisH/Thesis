from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .io import (
    discover_cases,
    image_path,
    label_path,
    load_ras,
    read_json,
    require_numpy,
    stable_hash,
    write_json,
)
from .logger import get_logger

logger = get_logger(__name__)


def robust_normalize(image, clip_z: float = 5.0, epsilon: float = 1e-6):
    """在非零脑区做 z-score，并裁剪缩放到 [-1, 1]。"""
    np = require_numpy()
    image = np.asarray(image, dtype=np.float32)
    foreground = np.isfinite(image) & (np.abs(image) > epsilon)
    if np.count_nonzero(foreground) < 32:
        foreground = np.isfinite(image)
    values = image[foreground]
    if values.size == 0:
        raise ValueError("image contains no finite foreground")
    mean = float(values.mean())
    std = float(values.std())
    if not math.isfinite(std) or std <= epsilon:
        raise ValueError("image foreground has near-zero standard deviation")
    normalized = np.clip((image - mean) / std, -clip_z, clip_z) / clip_z
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32), {
        "mean": mean,
        "std": std,
        "clip_z": float(clip_z),
        "foreground_voxels": int(values.size),
    }


def denormalize_image(image, metadata: dict[str, Any]):
    """使用保存的均值、标准差和裁剪尺度还原影像强度。"""
    np = require_numpy()
    return (
        np.asarray(image, dtype=np.float32)
        * float(metadata["clip_z"])
        * float(metadata["std"])
        + float(metadata["mean"])
    )


def centered_crop_with_padding(array, center, shape, *, pad_mode: str, pad_value: float = 0):
    """围绕中心裁剪固定大小数组，并在越界处按规则填充。"""
    np = require_numpy()
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
    """提取三维连通域及其体积、质心和包围盒信息。"""
    np = require_numpy()
    try:
        from scipy.ndimage import label
    except ImportError as exc:
        raise RuntimeError("SciPy is required for connected component extraction") from exc

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


def lesion_histogram(image, mask, bins: int = 16):
    """计算归一化病灶区域的定长强度直方图。"""
    np = require_numpy()
    values = np.asarray(image)[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        raise ValueError("cannot compute a histogram for an empty lesion")
    histogram, _ = np.histogram(values, bins=bins, range=(-1.0, 1.0))
    histogram = histogram.astype(np.float32)
    histogram /= max(float(histogram.sum()), 1.0)
    return histogram


def collect_case_statistics(dataset_dir: str | Path, cases: list[str], channel: int, label_id: int):
    """统计各病例的病灶体素数与连通分量数。"""
    logger.info("Collecting lesion statistics for %d cases (channel=%d, label=%d)", len(cases), channel, label_id)
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
    total_voxels = sum(s["lesion_voxels"] for s in statistics)
    total_components = sum(s["components"] for s in statistics)
    logger.info("Statistics collected: %d total lesion voxels, %d total components", total_voxels, total_components)
    return statistics


def stratified_split(
    statistics: list[dict[str, Any]],
    counts: dict[str, int],
    seed: int,
) -> dict[str, list[str]]:
    """Deterministically distribute adjacent severity ranks across all splits."""
    np = require_numpy()
    names = ("train", "val", "test")
    requested = {name: int(counts[name]) for name in names}
    if sum(requested.values()) != len(statistics):
        raise ValueError("split counts must equal the discovered patient count")

    ordered = sorted(
        statistics,
        key=lambda item: (item["lesion_voxels"], item["components"], item["case_id"]),
    )
    rng = np.random.default_rng(seed)
    # Shuffle only within local severity strata, preserving global balance.
    strata = np.array_split(np.asarray(ordered, dtype=object), min(8, len(ordered)))
    ranked: list[dict[str, Any]] = []
    for stratum in strata:
        items = list(stratum)
        rng.shuffle(items)
        ranked.extend(items)

    split: dict[str, list[str]] = {name: [] for name in names}
    remaining = requested.copy()
    for index, item in enumerate(ranked):
        candidates = [name for name in names if remaining[name] > 0]
        # Largest remaining fraction wins; rotate ties to avoid severity clumps.
        name = max(
            candidates,
            key=lambda value: (
                remaining[value] / max(requested[value], 1),
                -((names.index(value) - index) % len(names)),
            ),
        )
        split[name].append(item["case_id"])
        remaining[name] -= 1
    for name in names:
        split[name].sort()
    return split


def _valid_component(record: dict[str, Any], patch_size, margin: int, min_voxels: int) -> bool:
    """判断病灶能否满足最小体积和 patch 边距约束。"""
    maximum_bbox = [int(size) - 2 * margin for size in patch_size]
    return (
        int(record["voxels"]) >= min_voxels
        and all(int(actual) <= allowed for actual, allowed in zip(record["bbox_shape"], maximum_bbox))
    )


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    """生成患者划分、病灶 patch、直方图库和位置先验。"""
    np = require_numpy()
    data_cfg = config["data"]
    norm_cfg = config["normalization"]
    dataset_dir = Path(data_cfg["source_dataset"])
    prepared_dir = Path(data_cfg["prepared_dir"])
    patches_dir = prepared_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting data preparation ===")
    logger.info("Source dataset: %s", dataset_dir)
    logger.info("Output directory: %s", prepared_dir)

    channel = int(data_cfg["channel"])
    label_id = int(data_cfg["label_id"])
    patch_size = tuple(int(value) for value in data_cfg["patch_size"])
    cases = discover_cases(dataset_dir, channel)
    logger.info("Discovered %d cases", len(cases))

    statistics = collect_case_statistics(dataset_dir, cases, channel, label_id)
    split = stratified_split(statistics, data_cfg["split_counts"], int(config["seed"]))
    logger.info("Split: train=%d, val=%d, test=%d", len(split["train"]), len(split["val"]), len(split["test"]))

    split_document = {
        "seed": int(config["seed"]),
        "counts": {name: len(values) for name, values in split.items()},
        "cases": split,
        "statistics": statistics,
    }
    split_document["hash"] = stable_hash(split_document)
    write_json(prepared_dir / "split.json", split_document)
    logger.info("Split saved to %s", prepared_dir / "split.json")

    case_to_split = {
        case_id: split_name for split_name, values in split.items() for case_id in values
    }
    entries: list[dict[str, Any]] = []
    histograms: list[list[float]] = []
    rejected: list[dict[str, Any]] = []

    logger.info("Extracting lesion patches (patch_size=%s, margin=%d, min_voxels=%d)...",
                 patch_size, data_cfg["patch_margin"], data_cfg["min_component_voxels"])
    for case_id in tqdm(cases, desc="Extracting lesion patches", unit="case"):
        image, image_obj = load_ras(image_path(dataset_dir, case_id, channel))
        label_array, label_obj = load_ras(label_path(dataset_dir, case_id), label=True)
        if image.shape != label_array.shape:
            raise ValueError(f"{case_id}: image and label shapes differ")
        if not np.allclose(image_obj.affine, label_obj.affine, atol=1e-4):
            raise ValueError(f"{case_id}: image and label affines differ after RAS conversion")

        normalized, norm_metadata = robust_normalize(
            image,
            clip_z=float(norm_cfg["clip_z"]),
            epsilon=float(norm_cfg["foreground_epsilon"]),
        )
        binary = label_array == label_id
        components, records = component_records(binary)
        for record in records:
            if not _valid_component(
                record,
                patch_size,
                int(data_cfg["patch_margin"]),
                int(data_cfg["min_component_voxels"]),
            ):
                rejected.append({"case_id": case_id, **record, "reason": "size_or_margin"})
                continue
            component_mask = components == int(record["component_id"])
            center = [int(round(value)) for value in record["centroid"]]
            image_patch, crop = centered_crop_with_padding(
                normalized, center, patch_size, pad_mode="reflect"
            )
            mask_patch, _ = centered_crop_with_padding(
                component_mask, center, patch_size, pad_mode="constant", pad_value=0
            )
            if any(np.any(np.take(mask_patch, (0, -1), axis=axis)) for axis in range(3)):
                rejected.append({"case_id": case_id, **record, "reason": "touches_patch_edge"})
                continue

            histogram = lesion_histogram(
                image_patch, mask_patch, bins=int(norm_cfg["histogram_bins"])
            )
            patch_id = f"{case_id}_lesion{int(record['component_id']):03d}"
            patch_path = patches_dir / f"{patch_id}.npz"
            np.savez_compressed(
                patch_path,
                image=image_patch.astype(np.float32)[None],
                mask=mask_patch.astype(np.uint8)[None],
                histogram=histogram,
            )
            entry = {
                "patch_id": patch_id,
                "case_id": case_id,
                "split": case_to_split[case_id],
                "component_id": int(record["component_id"]),
                "voxels": int(record["voxels"]),
                "centroid_ras_voxel": record["centroid"],
                "centroid_fraction": [
                    float(value / max(size - 1, 1))
                    for value, size in zip(record["centroid"], image.shape)
                ],
                "crop": crop,
                "patch": str(patch_path.relative_to(prepared_dir)),
                "normalization": norm_metadata,
                "source_image": str(image_path(dataset_dir, case_id, channel)),
                "source_label": str(label_path(dataset_dir, case_id)),
                "affine": image_obj.affine.tolist(),
            }
            entries.append(entry)
            if case_to_split[case_id] == "train":
                histograms.append(histogram.tolist())

    if not entries:
        raise RuntimeError("no eligible lesion patches were produced")
    if not histograms:
        raise RuntimeError("training split produced no eligible lesion histograms")
    manifest = {
        "version": 1,
        "source_dataset": str(dataset_dir),
        "channel": channel,
        "label_id": label_id,
        "patch_size": list(patch_size),
        "split_hash": split_document["hash"],
        "entries": entries,
        "rejected": rejected,
    }
    manifest["hash"] = stable_hash(manifest)
    write_json(prepared_dir / "manifest.json", manifest)
    np.save(prepared_dir / "train_histograms.npy", np.asarray(histograms, dtype=np.float32))
    write_json(
        prepared_dir / "position_prior.json",
        {
            "coordinate_system": "RAS fractional voxel coordinates",
            "registered_dir": data_cfg.get("registered_dir"),
            "centers": [
                entry["centroid_fraction"] for entry in entries if entry["split"] == "train"
            ],
        },
    )
    logger.info("Manifest: %d entries, %d rejected", len(entries), len(rejected))
    logger.info("Training histograms: %d", len(histograms))
    logger.info("Artifacts saved to %s", prepared_dir)
    logger.info("=== Data preparation complete ===")
    return {
        "cases": len(cases),
        "patches": len(entries),
        "training_histograms": len(histograms),
        "rejected": len(rejected),
        "split": split_document["counts"],
        "manifest": str(prepared_dir / "manifest.json"),
    }


def _affine_zoom_pair(image, mask, scale: float):
    """以相同中心缩放影像与掩膜，并使用各自合适的插值。"""
    np = require_numpy()
    try:
        from scipy.ndimage import affine_transform
    except ImportError as exc:
        raise RuntimeError("SciPy is required for augmentation") from exc
    center = (np.asarray(image.shape, dtype=np.float64) - 1.0) / 2.0
    matrix = np.eye(3, dtype=np.float64) / scale
    offset = center - matrix @ center
    return (
        affine_transform(image, matrix, offset=offset, order=1, mode="reflect"),
        affine_transform(mask.astype(np.float32), matrix, offset=offset, order=0, mode="constant") > 0.5,
    )


def augment_pair(image, mask, config: dict[str, Any], rng):
    """同步执行翻转、旋转、缩放和弹性形变。"""
    np = require_numpy()
    try:
        from scipy.ndimage import gaussian_filter, map_coordinates, rotate
    except ImportError as exc:
        raise RuntimeError("SciPy is required for augmentation") from exc

    original_image = np.asarray(image, dtype=np.float32)
    original_mask = np.asarray(mask, dtype=bool)
    for _ in range(int(config.get("max_attempts", 8))):
        transformed_image = original_image.copy()
        transformed_mask = original_mask.copy()
        for axis in range(3):
            if rng.random() < 0.5:
                transformed_image = np.flip(transformed_image, axis=axis).copy()
                transformed_mask = np.flip(transformed_mask, axis=axis).copy()
        axes = ((0, 1), (0, 2), (1, 2))[int(rng.integers(0, 3))]
        angle = float(rng.uniform(-config["max_rotation_deg"], config["max_rotation_deg"]))
        transformed_image = rotate(
            transformed_image, angle, axes=axes, reshape=False, order=1, mode="reflect"
        )
        transformed_mask = rotate(
            transformed_mask.astype(np.float32),
            angle,
            axes=axes,
            reshape=False,
            order=0,
            mode="constant",
        ) > 0.5
        scale = float(rng.uniform(*config["scale_range"]))
        transformed_image, transformed_mask = _affine_zoom_pair(
            transformed_image, transformed_mask, scale
        )

        alpha = float(config.get("elastic_alpha", 0.0))
        if alpha > 0:
            sigma = float(config.get("elastic_sigma", 4.0))
            coordinates = np.meshgrid(
                *[np.arange(size, dtype=np.float32) for size in transformed_image.shape],
                indexing="ij",
            )
            displaced = [
                coordinate
                + gaussian_filter(
                    rng.normal(size=transformed_image.shape), sigma=sigma, mode="reflect"
                )
                * alpha
                for coordinate in coordinates
            ]
            transformed_image = map_coordinates(
                transformed_image, displaced, order=1, mode="reflect"
            )
            transformed_mask = (
                map_coordinates(
                    transformed_mask.astype(np.float32),
                    displaced,
                    order=0,
                    mode="constant",
                )
                > 0.5
            )
        if transformed_mask.sum() < 8:
            continue
        if any(
            np.any(np.take(transformed_mask, (0, -1), axis=axis)) for axis in range(3)
        ):
            continue
        return transformed_image.astype(np.float32), transformed_mask
    raise RuntimeError("augmentation failed to produce a valid lesion mask")


class MeningitisPatchDataset:
    """PyTorch-compatible dataset with patient-split enforcement."""

    def __init__(
        self,
        prepared_dir: str | Path,
        split: str,
        *,
        augmentation: dict[str, Any] | None = None,
        seed: int = 42,
    ):
        """载入指定患者划分的 manifest 条目与增强设置。"""
        self.prepared_dir = Path(prepared_dir)
        manifest = read_json(self.prepared_dir / "manifest.json")
        self.entries = [entry for entry in manifest["entries"] if entry["split"] == split]
        if not self.entries:
            raise ValueError(f"manifest has no entries for split={split}")
        self.augmentation = augmentation if augmentation and augmentation.get("enabled") else None
        self.seed = int(seed)
        logger.info("MeningitisPatchDataset: split=%s, entries=%d, augmentation=%s",
                     split, len(self.entries), bool(self.augmentation))

    def __len__(self) -> int:
        """返回当前划分中的可用病灶 patch 数量。"""
        return len(self.entries)

    def __getitem__(self, index: int):
        """读取一个 patch，按需增强并转换为 PyTorch 张量。"""
        np = require_numpy()
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for training") from exc
        entry = self.entries[index]
        with np.load(self.prepared_dir / entry["patch"]) as sample:
            image = sample["image"][0]
            mask = sample["mask"][0].astype(bool)
            histogram = sample["histogram"]
        if self.augmentation:
            # Worker seed is injected by torch; index keeps direct access reproducible.
            rng = np.random.default_rng(self.seed + index + int(torch.initial_seed() % 2**31))
            image, mask = augment_pair(image, mask, self.augmentation, rng)
            histogram = lesion_histogram(image, mask, bins=len(histogram))
        return {
            "image": torch.from_numpy(image[None].copy()).float(),
            "mask": torch.from_numpy(mask[None].copy()),
            "histogram": torch.from_numpy(np.asarray(histogram).copy()).float(),
            "case_id": entry["case_id"],
            "component_id": entry["component_id"],
        }
