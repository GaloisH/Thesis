from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..io import (
    discover_cases,
    image_path,
    label_path,
    load_ras,
    require_numpy,
    stable_hash,
    write_json,
)
from ..logger import get_logger
from .components import (
    centered_crop_with_padding,
    collect_case_statistics,
    component_records,
)
from .normalization import lesion_histogram, robust_normalize

logger = get_logger(__name__)


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


def _valid_component(
    record: dict[str, Any],
    patch_size,
    margin: int,
    min_voxels: int,
) -> bool:
    maximum_bbox = [int(size) - 2 * margin for size in patch_size]
    return int(record["voxels"]) >= min_voxels and all(
        int(actual) <= allowed
        for actual, allowed in zip(record["bbox_shape"], maximum_bbox)
    )


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    """Create patient splits, lesion patches, histograms, and a position prior."""
    np = require_numpy()
    data_cfg = config["data"]
    norm_cfg = config["normalization"]
    dataset_dir = Path(data_cfg["source_dataset"])
    prepared_dir = Path(data_cfg["prepared_dir"])
    patches_dir = prepared_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    channel = int(data_cfg["channel"])
    label_id = int(data_cfg["label_id"])
    patch_size = tuple(int(value) for value in data_cfg["patch_size"])
    cases = discover_cases(dataset_dir, channel)
    statistics = collect_case_statistics(dataset_dir, cases, channel, label_id)
    split = stratified_split(statistics, data_cfg["split_counts"], int(config["seed"]))
    split_document = {
        "seed": int(config["seed"]),
        "counts": {name: len(values) for name, values in split.items()},
        "cases": split,
        "statistics": statistics,
    }
    split_document["hash"] = stable_hash(split_document)
    write_json(prepared_dir / "split.json", split_document)

    case_to_split = {
        case_id: split_name
        for split_name, values in split.items()
        for case_id in values
    }
    entries: list[dict[str, Any]] = []
    histograms: list[list[float]] = []
    rejected: list[dict[str, Any]] = []
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
        components, records = component_records(label_array == label_id)
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
                component_mask,
                center,
                patch_size,
                pad_mode="constant",
                pad_value=0,
            )
            if any(np.any(np.take(mask_patch, (0, -1), axis=axis)) for axis in range(3)):
                rejected.append(
                    {"case_id": case_id, **record, "reason": "touches_patch_edge"}
                )
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
                entry["centroid_fraction"]
                for entry in entries
                if entry["split"] == "train"
            ],
        },
    )
    logger.info(
        "Data preparation complete: cases=%d, patches=%d, rejected=%d",
        len(cases),
        len(entries),
        len(rejected),
    )
    return {
        "cases": len(cases),
        "patches": len(entries),
        "training_histograms": len(histograms),
        "rejected": len(rejected),
        "split": split_document["counts"],
        "manifest": str(prepared_dir / "manifest.json"),
    }
