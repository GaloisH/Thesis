from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..data import denormalize_image, robust_normalize
from ..io import (
    require_numpy,
    restore_ras_to_source,
    save_like,
    stable_hash,
    write_json,
)
from ..synthesis import qc_patch, roi_from_mask, sample_composite_patch, sample_histogram
from .inputs import _mask_from_prepared_entry, load_aligned_ras
from .plots import (
    _plot_comparison,
    _plot_qc,
    embed_patch,
    masked_absolute_difference,
    render_figures,
)
from .reporting import save_patch_nifti


def _process_case(
    config: dict[str, Any],
    model,
    checkpoint,
    device,
    *,
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    case_id: str,
    seed: int,
    histogram_library,
    prepared_entry=None,
    prepared_dir: Path | None = None,
):
    """Generate and visualize one fixed-mask synthesis case."""
    np = require_numpy()
    image, mask, image_ras, image_source = load_aligned_ras(image_path, mask_path)
    if prepared_entry is not None:
        if prepared_dir is None:
            raise ValueError("prepared_dir is required for a prepared component")
        mask = _mask_from_prepared_entry(image.shape, prepared_dir, prepared_entry)
    patch_shape = tuple(int(value) for value in config["data"]["patch_size"])
    margin = int(config["data"].get("patch_margin", 0))
    roi, roi_meta = roi_from_mask(mask, patch_shape, margin=margin)
    mask_patch = mask[roi]

    normalized, normalization = robust_normalize(
        image,
        float(config["normalization"]["clip_z"]),
        float(config["normalization"]["foreground_epsilon"]),
    )
    rng = np.random.default_rng(int(seed))
    histogram = sample_histogram(
        histogram_library,
        rng,
        float(config["synthesis"]["histogram_jitter"]),
    )
    generated_normalized, composite_normalized = sample_composite_patch(
        model,
        normalized[roi],
        mask_patch,
        histogram,
        device=device,
        seed=seed,
        brightness_margin=float(config["synthesis"].get("brightness_margin", 0.1)),
        brightness_transition_voxels=float(
            config["synthesis"].get("brightness_transition_voxels", 3.0)
        ),
    )
    qc = qc_patch(
        normalized[roi],
        generated_normalized,
        composite_normalized,
        mask_patch,
        config["synthesis"],
    )
    if not qc["passed"] and not bool(
        config["visualization"].get("save_failed_qc", True)
    ):
        raise RuntimeError(
            "synthesis QC failed and visualization.save_failed_qc is false: "
            + ", ".join(qc["failures"])
        )

    generated_raw = denormalize_image(generated_normalized, normalization)
    synthetic = image.copy()
    synthetic_roi = synthetic[roi]
    synthetic_roi[mask_patch] = generated_raw[mask_patch]
    synthetic[roi] = synthetic_roi
    if not np.array_equal(synthetic[~mask], image[~mask]):
        raise AssertionError("fixed-mask synthesis changed voxels outside the mask")
    difference = masked_absolute_difference(image, synthetic, mask)
    if np.any(difference[~mask] != 0):
        raise AssertionError("difference map is non-zero outside the mask")

    output_dir.mkdir(parents=True, exist_ok=True)
    original_output = output_dir / "original.nii.gz"
    synthetic_output = output_dir / "synthetic.nii.gz"
    mask_output = output_dir / "inserted_mask.nii.gz"
    patch_output = output_dir / "generated_patch.nii.gz"
    keep_nifti = bool(config["visualization"].get("keep_intermediate_nifti", True))
    if keep_nifti:
        shutil.copy2(image_path, original_output)
    synthetic_native = restore_ras_to_source(synthetic, image_source)
    mask_native = restore_ras_to_source(mask.astype(np.uint8), image_source)
    save_like(synthetic_native, image_source, synthetic_output, dtype=np.float32)
    save_like(mask_native, image_source, mask_output, dtype=np.uint8)
    if keep_nifti:
        save_patch_nifti(generated_raw, image_ras, roi_meta["start"], patch_output)

    generated_full = embed_patch(image.shape, generated_raw, roi)
    cfg = config["visualization"]
    paths, center, window = render_figures(
        image,
        generated_full,
        synthetic,
        mask,
        image_ras.header.get_zooms()[:3],
        roi_meta,
        output_dir,
        cfg,
    )
    _plot_qc(
        image[roi],
        generated_raw,
        synthetic[roi],
        mask_patch,
        qc,
        paths["qc"],
        cfg,
    )
    _plot_comparison(
        image,
        generated_full,
        synthetic,
        mask,
        center,
        paths["comparison"],
        cfg,
        window,
    )

    metadata = {
        "case_id": case_id,
        "input_image": str(image_path.resolve()),
        "input_mask": str(mask_path.resolve()),
        "mask_source": (
            {
                "mode": "largest_prepared_component",
                "patch_id": prepared_entry["patch_id"],
                "component_id": int(prepared_entry["component_id"]),
                "voxels": int(prepared_entry["voxels"]),
            }
            if prepared_entry is not None
            else {"mode": "explicit_full_volume_mask"}
        ),
        "checkpoint": str(config["synthesis"]["checkpoint"]),
        "checkpoint_step": int(checkpoint.get("global_step", -1)),
        "seed": int(seed),
        "slice_center_ras_voxel": [int(value) for value in center],
        "roi_ras_voxel": roi_meta,
        "normalization": normalization,
        "histogram": histogram.tolist(),
        "qc": qc,
        "geometry": {
            "shape": [int(value) for value in image_source.shape],
            "spacing": [float(value) for value in image_source.header.get_zooms()[:3]],
            "affine": image_source.affine.tolist(),
        },
        "outputs": {
            "original": str(original_output) if keep_nifti else None,
            "generated_patch": str(patch_output) if keep_nifti else None,
            "synthetic": str(synthetic_output),
            "inserted_mask": str(mask_output),
            "figures": {key: str(value) for key, value in paths.items()},
        },
    }
    metadata["hash"] = stable_hash(metadata)
    write_json(output_dir / "metadata.json", metadata)
    return metadata

