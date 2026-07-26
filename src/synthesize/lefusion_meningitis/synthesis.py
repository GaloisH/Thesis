from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from .data import (
    _affine_zoom_pair,
    centered_crop_with_padding,
    denormalize_image,
    robust_normalize,
)
from .io import (
    image_path,
    label_path,
    load_ras_with_source,
    read_json,
    require_numpy,
    restore_ras_to_source,
    save_like,
    stable_hash,
    write_json,
)
from .logger import get_logger
from .model import load_model_checkpoint, require_torch

logger = get_logger(__name__)


def transform_donor_mask(mask, rng, rotation_deg: float, scale_range):
    """对真实供体掩膜执行轻度随机旋转和缩放。"""
    np = require_numpy()
    try:
        from scipy.ndimage import rotate
    except ImportError as exc:
        raise RuntimeError("SciPy is required for mask transformation") from exc
    result = np.asarray(mask, dtype=bool)
    axes = ((0, 1), (0, 2), (1, 2))[int(rng.integers(0, 3))]
    angle = float(rng.uniform(-rotation_deg, rotation_deg))
    result = (
        rotate(
            result.astype(np.float32),
            angle,
            axes=axes,
            reshape=False,
            order=0,
            mode="constant",
        )
        > 0.5
    )
    _, result = _affine_zoom_pair(
        np.zeros(result.shape, dtype=np.float32),
        result,
        float(rng.uniform(*scale_range)),
    )
    return result


def choose_candidate(
    image,
    existing_label,
    donor_mask,
    position_centers,
    rng,
    *,
    protected_dilation: int,
    max_attempts: int,
):
    """按位置先验寻找位于脑区且避开已有病灶的放置位置。"""
    np = require_numpy()
    try:
        from scipy.ndimage import binary_dilation, binary_fill_holes, label
    except ImportError as exc:
        raise RuntimeError("SciPy is required for lesion placement") from exc

    finite = np.isfinite(image)
    nonzero = finite & (np.abs(image) > 1e-6)
    components, count = label(nonzero)
    if count:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        brain = binary_fill_holes(components == int(sizes.argmax()))
    else:
        brain = finite
    protected = binary_dilation(existing_label > 0, iterations=protected_dilation)
    patch_shape = np.asarray(donor_mask.shape)
    half = patch_shape // 2
    centers = np.asarray(position_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("position prior must contain Nx3 centers")

    for _ in range(max_attempts):
        fraction = centers[int(rng.integers(0, len(centers)))].copy()
        fraction += rng.normal(0.0, 0.02, size=3)
        fraction = np.clip(fraction, 0.0, 1.0)
        center = np.rint(fraction * (np.asarray(image.shape) - 1)).astype(np.int64)
        start = center - half
        end = start + patch_shape
        if np.any(start < 0) or np.any(end > np.asarray(image.shape)):
            continue
        roi = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
        if not np.all(brain[roi][donor_mask]):
            continue
        if np.any(protected[roi][donor_mask]):
            continue
        return tuple(int(value) for value in center), roi
    raise RuntimeError("no anatomically valid lesion placement found")


def sample_histogram(histograms, rng, jitter: float):
    """从训练直方图库采样条件并施加归一化小扰动。"""
    np = require_numpy()
    histogram = np.asarray(
        histograms[int(rng.integers(0, len(histograms)))], dtype=np.float32
    ).copy()
    if jitter > 0:
        histogram += rng.normal(0.0, jitter, size=histogram.shape).astype(np.float32)
    histogram = np.clip(histogram, 0.0, None)
    total = float(histogram.sum())
    if total <= 0:
        raise ValueError("histogram perturbation produced an empty distribution")
    return histogram / total


def roi_from_mask(mask, patch_shape, *, margin: int = 0):
    """Return a fixed-size ROI centered on a full-volume lesion mask.

    The lesion must fit inside the patch with the requested margin. Padding is
    deliberately not allowed because a padded image patch would change the
    anatomical background seen by the model.
    """
    np = require_numpy()
    lesion = np.asarray(mask, dtype=bool)
    if lesion.ndim != 3:
        raise ValueError(f"lesion mask must be 3D, got shape {lesion.shape}")
    coordinates = np.argwhere(lesion)
    if coordinates.size == 0:
        raise ValueError("lesion mask is empty")

    patch = np.asarray(patch_shape, dtype=np.int64)
    if patch.shape != (3,) or np.any(patch <= 0):
        raise ValueError(f"patch shape must contain three positive values: {patch_shape}")
    margin = int(margin)
    if margin < 0 or np.any(patch <= 2 * margin):
        raise ValueError("mask margin is incompatible with patch shape")

    bbox_min = coordinates.min(axis=0)
    bbox_max = coordinates.max(axis=0) + 1
    bbox_shape = bbox_max - bbox_min
    available = patch - 2 * margin
    if np.any(bbox_shape > available):
        raise ValueError(
            "lesion mask does not fit inside the model patch with margin: "
            f"bbox={tuple(int(v) for v in bbox_shape)}, "
            f"available={tuple(int(v) for v in available)}"
        )

    center = np.floor((bbox_min + bbox_max - 1) / 2.0).astype(np.int64)
    start = center - patch // 2
    end = start + patch
    if np.any(start < 0) or np.any(end > np.asarray(lesion.shape)):
        raise ValueError(
            "lesion is too close to the image edge for an unpadded model patch"
        )
    if np.any(bbox_min < start + margin) or np.any(bbox_max > end - margin):
        raise ValueError(
            "lesion cannot be centered in the model patch with the requested margin"
        )
    roi = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
    return roi, {
        "center": [int(v) for v in center],
        "start": [int(v) for v in start],
        "end": [int(v) for v in end],
        "bbox_min": [int(v) for v in bbox_min],
        "bbox_max": [int(v) for v in bbox_max],
        "bbox_shape": [int(v) for v in bbox_shape],
        "margin": margin,
    }


def sample_composite_patch(
    model,
    background,
    mask,
    histogram,
    *,
    device,
    seed: int,
    brightness_margin: float = 0.1,
    brightness_transition_voxels: float = 3.0,
):
    """Generate a lesion patch and hard-composite it into its real background."""
    np = require_numpy()
    torch = require_torch()
    background = np.asarray(background, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    histogram = np.asarray(histogram, dtype=np.float32)
    if background.shape != mask.shape:
        raise ValueError("background and mask patch shapes differ")
    if not mask.any():
        raise ValueError("synthesis mask is empty")

    background_tensor = torch.from_numpy(background[None, None]).to(device)
    mask_tensor = torch.from_numpy(mask[None, None]).to(device)
    histogram_tensor = torch.from_numpy(histogram[None]).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    generated_tensor = model.sample_patch(
        background_tensor, mask_tensor, histogram_tensor, generator=generator
    )
    generated = generated_tensor[0, 0].detach().cpu().numpy()
    generated = brighten_lesion_interior(
        background,
        generated,
        mask,
        margin=brightness_margin,
        transition_voxels=brightness_transition_voxels,
    )
    return generated, hard_composite(background, generated, mask)


def brighten_lesion_interior(
    background,
    generated,
    mask,
    *,
    margin: float = 0.1,
    transition_voxels: float = 3.0,
):
    """Raise lesion intensity progressively from its boundary toward its center."""
    np = require_numpy()
    try:
        from scipy.ndimage import binary_dilation, distance_transform_edt
    except ImportError as exc:
        raise RuntimeError("SciPy is required for lesion brightening") from exc

    background = np.asarray(background, dtype=np.float32)
    adjusted = np.asarray(generated, dtype=np.float32).copy()
    mask = np.asarray(mask, dtype=bool)
    if background.shape != adjusted.shape or mask.shape != background.shape:
        raise ValueError("brightening inputs have incompatible shapes")
    if not mask.any():
        raise ValueError("brightening mask is empty")
    if margin < 0:
        raise ValueError("brightness margin must be non-negative")
    if transition_voxels <= 0:
        raise ValueError("brightness transition must be positive")

    ring_width = max(1, int(np.ceil(transition_voxels)))
    ring = binary_dilation(mask, iterations=ring_width) & ~mask
    if not ring.any():
        return adjusted

    background_level = float(np.percentile(background[ring], 90))
    lesion_level = float(np.percentile(adjusted[mask], 25))
    offset = max(0.0, background_level + float(margin) - lesion_level)
    if offset == 0:
        return adjusted

    distance = distance_transform_edt(mask)
    weight = np.clip(distance / float(transition_voxels), 0.0, 1.0)
    adjusted[mask] = np.clip(
        adjusted[mask] + offset * weight[mask],
        -1.0,
        1.0,
    )
    return adjusted


def hard_composite(background, generated, mask):
    """仅替换掩膜内部体素，严格保留掩膜外背景。"""
    np = require_numpy()
    background = np.asarray(background)
    generated = np.asarray(generated)
    mask = np.asarray(mask, dtype=bool)
    if background.shape != generated.shape or mask.shape != background.shape:
        raise ValueError("composite inputs have incompatible shapes")
    result = background.copy()
    result[mask] = generated[mask]
    return result


def qc_patch(background, generated, composite, mask, config: dict[str, Any]):
    """检查体积、边缘、强度、背景不变性和边界连续性。"""
    np = require_numpy()
    try:
        from scipy.ndimage import binary_dilation
    except ImportError as exc:
        raise RuntimeError("SciPy is required for synthesis QC") from exc

    failures: list[str] = []
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 8:
        failures.append("mask_too_small")
    if any(np.any(np.take(mask, (0, -1), axis=axis)) for axis in range(3)):
        failures.append("mask_touches_patch_edge")
    if not np.array_equal(composite[~mask], background[~mask]):
        failures.append("background_changed")
    if not np.all(np.isfinite(composite)):
        failures.append("non_finite")
    if mask.any() and float(np.max(np.abs(generated[mask]))) > float(
        config["intensity_z_limit"]
    ):
        failures.append("generated_intensity_outlier")
    inner = mask & ~binary_dilation(~mask, iterations=1)
    outer = binary_dilation(mask, iterations=1) & ~mask
    if inner.any() and outer.any():
        boundary_jump = abs(float(composite[inner].mean() - composite[outer].mean()))
    else:
        boundary_jump = float("inf")
    if boundary_jump > float(config["max_boundary_jump_z"]):
        failures.append("boundary_jump")
    return {
        "passed": not failures,
        "failures": failures,
        "mask_voxels": int(mask.sum()),
        "background_exact": bool(np.array_equal(composite[~mask], background[~mask])),
        "boundary_jump": boundary_jump,
    }


def synthesize(config: dict[str, Any]) -> dict[str, Any]:
    """Sample lesions into target cases and write full-volume NIfTI training pairs."""
    np = require_numpy()
    torch = require_torch()
    synthesis_cfg = config["synthesis"]
    data_cfg = config["data"]
    prepared_dir = Path(data_cfg["prepared_dir"])
    output_dir = Path(synthesis_cfg["output_dir"])
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    masks_dir = output_dir / "masks"
    metadata_dir = output_dir / "metadata"
    for directory in (images_dir, labels_dir, masks_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting synthesis ===")
    logger.info("Output directory: %s", output_dir)
    logger.info("Samples per case: %d", synthesis_cfg["num_per_case"])

    manifest = read_json(prepared_dir / "manifest.json")
    split = read_json(prepared_dir / "split.json")
    prior = read_json(prepared_dir / "position_prior.json")
    histogram_library = np.load(prepared_dir / "train_histograms.npy").copy()
    train_entries = [entry for entry in manifest["entries"] if entry["split"] == "train"]
    target_split = str(synthesis_cfg.get("split", "train"))
    targets = list(split["cases"][target_split])
    if not train_entries:
        raise RuntimeError("training manifest contains no donor lesions")

    logger.info("Target cases: %d, donor lesions: %d", len(targets), len(train_entries))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    model, checkpoint = load_model_checkpoint(
        synthesis_cfg["checkpoint"], config["model"], device
    )
    base_seed = int(config["seed"])
    records: list[dict[str, Any]] = []
    rejected = 0
    num_per_case = int(synthesis_cfg["num_per_case"])

    for target_index, target_case in enumerate(
        tqdm(targets, desc="Synthesizing", unit="case")
    ):
        for sample_index in range(num_per_case):
            seed = base_seed + target_index * 100_003 + sample_index
            rng = np.random.default_rng(seed)
            donor_candidates = [
                entry for entry in train_entries if entry["case_id"] != target_case
            ]
            if not donor_candidates:
                donor_candidates = train_entries
            donor = donor_candidates[int(rng.integers(0, len(donor_candidates)))]
            with np.load(prepared_dir / donor["patch"]) as donor_sample:
                donor_mask = donor_sample["mask"][0].astype(bool)
            donor_mask = transform_donor_mask(
                donor_mask,
                rng,
                float(synthesis_cfg["mask_rotation_deg"]),
                synthesis_cfg["mask_scale_range"],
            )

            raw_image, ras_image, source_image = load_ras_with_source(
                image_path(data_cfg["source_dataset"], target_case, int(data_cfg["channel"]))
            )
            raw_label, _, source_label = load_ras_with_source(
                label_path(data_cfg["source_dataset"], target_case), label=True
            )
            normalized, norm_metadata = robust_normalize(
                raw_image,
                float(config["normalization"]["clip_z"]),
                float(config["normalization"]["foreground_epsilon"]),
            )
            try:
                center, roi = choose_candidate(
                    raw_image,
                    raw_label,
                    donor_mask,
                    prior["centers"],
                    rng,
                    protected_dilation=int(synthesis_cfg["protected_dilation"]),
                    max_attempts=int(synthesis_cfg["max_placement_attempts"]),
                )
            except RuntimeError:
                rejected += 1
                continue

            background = normalized[roi].astype(np.float32)
            histogram = sample_histogram(
                histogram_library, rng, float(synthesis_cfg["histogram_jitter"])
            )
            generated, composite = sample_composite_patch(
                model,
                background,
                donor_mask,
                histogram,
                device=device,
                seed=seed,
                brightness_margin=float(synthesis_cfg.get("brightness_margin", 0.1)),
                brightness_transition_voxels=float(
                    synthesis_cfg.get("brightness_transition_voxels", 3.0)
                ),
            )
            qc = qc_patch(background, generated, composite, donor_mask, synthesis_cfg)
            if not qc["passed"]:
                rejected += 1
                continue

            synthetic_normalized = normalized.copy()
            synthetic_normalized[roi] = composite
            synthetic_raw = denormalize_image(synthetic_normalized, norm_metadata)
            inserted_mask = np.zeros(raw_label.shape, dtype=np.uint8)
            inserted_mask[roi][donor_mask] = 1
            union_label = np.maximum(raw_label.astype(np.uint8), inserted_mask)

            sample_id = f"{target_case}_syn{sample_index:03d}"
            synthetic_native = restore_ras_to_source(synthetic_raw, source_image)
            label_native = restore_ras_to_source(union_label, source_label)
            mask_native = restore_ras_to_source(inserted_mask, source_label)
            image_output = images_dir / f"{sample_id}_0000.nii.gz"
            label_output = labels_dir / f"{sample_id}.nii.gz"
            mask_output = masks_dir / f"{sample_id}.nii.gz"
            save_like(synthetic_native, source_image, image_output, dtype=np.float32)
            save_like(label_native, source_label, label_output, dtype=np.uint8)
            save_like(mask_native, source_label, mask_output, dtype=np.uint8)

            metadata = {
                "sample_id": sample_id,
                "seed": seed,
                "source_case": donor["case_id"],
                "source_component": donor["component_id"],
                "target_case": target_case,
                "center_ras_voxel": list(center),
                "histogram": histogram.tolist(),
                "normalization": norm_metadata,
                "checkpoint": str(synthesis_cfg["checkpoint"]),
                "checkpoint_step": int(checkpoint.get("global_step", -1)),
                "manifest_hash": manifest["hash"],
                "split_hash": split["hash"],
                "qc": qc,
                "outputs": {
                    "image": str(image_output),
                    "label": str(label_output),
                    "inserted_mask": str(mask_output),
                },
            }
            metadata["hash"] = stable_hash(metadata)
            write_json(metadata_dir / f"{sample_id}.json", metadata)
            records.append(metadata)

    summary = {
        "requested": len(targets) * num_per_case,
        "accepted": len(records),
        "rejected": rejected,
        "qc_rate": len(records) / max(len(records) + rejected, 1),
        "records": [record["sample_id"] for record in records],
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    logger.info(
        "=== Synthesis complete: %d accepted, %d rejected (QC rate: %.1f%%) ===",
        summary["accepted"], summary["rejected"], summary["qc_rate"] * 100,
    )
    return summary
