"""Generate fixed-mask LeFusion-H samples and publication-ready figures."""

from __future__ import annotations

import csv
import html
import shutil
from pathlib import Path
from typing import Any

from .data import denormalize_image, robust_normalize
from .io import (
    image_path,
    label_path,
    read_json,
    require_nibabel,
    require_numpy,
    restore_ras_to_source,
    save_like,
    stable_hash,
    write_json,
)
from .logger import get_logger
from .model import load_model_checkpoint, require_torch
from .synthesis import (
    qc_patch,
    roi_from_mask,
    sample_composite_patch,
    sample_histogram,
)

logger = get_logger(__name__)


def _case_id(path: str | Path) -> str:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = Path(name).stem
    if len(name) > 5 and name[-5] == "_" and name[-4:].isdigit():
        name = name[:-5]
    return name


def _load_aligned_ras(image_path: str | Path, mask_path: str | Path):
    """Load an image/mask pair after strict source-space geometry validation."""
    np = require_numpy()
    nib = require_nibabel()
    image_source = nib.load(str(image_path))
    mask_source = nib.load(str(mask_path))
    if len(image_source.shape) != 3 or len(mask_source.shape) != 3:
        raise ValueError("image and lesion mask must both be 3D NIfTI volumes")
    if tuple(image_source.shape) != tuple(mask_source.shape):
        raise ValueError(
            f"image/mask shapes differ: {image_source.shape} != {mask_source.shape}"
        )
    if not np.allclose(image_source.affine, mask_source.affine, rtol=1e-5, atol=1e-4):
        raise ValueError("image and lesion mask affines differ")

    image_ras = nib.as_closest_canonical(image_source)
    mask_ras = nib.as_closest_canonical(mask_source)
    image = np.asarray(image_ras.dataobj, dtype=np.float32)
    mask = np.asarray(mask_ras.dataobj) > 0
    if image.shape != mask.shape:
        raise ValueError("canonical image/mask shapes differ")
    if not np.all(np.isfinite(image)):
        raise ValueError("image contains non-finite values")
    if not mask.any():
        raise ValueError("lesion mask is empty")
    return image, mask, image_ras, image_source


def masked_absolute_difference(before, after, mask):
    """Return an absolute difference map that is exactly zero outside the mask."""
    np = require_numpy()
    difference = np.zeros(np.asarray(before).shape, dtype=np.float32)
    lesion = np.asarray(mask, dtype=bool)
    difference[lesion] = np.abs(
        np.asarray(after, dtype=np.float32)[lesion]
        - np.asarray(before, dtype=np.float32)[lesion]
    )
    return difference


def _mask_from_prepared_entry(shape, prepared_dir: Path, entry):
    """Restore one prepared component mask into its full RAS voxel grid."""
    np = require_numpy()
    with np.load(prepared_dir / entry["patch"]) as sample:
        patch = np.asarray(sample["mask"][0], dtype=bool)
    crop = entry["crop"]
    start = np.asarray(crop["start"], dtype=np.int64)
    end = np.asarray(crop["end"], dtype=np.int64)
    before = np.asarray([item[0] for item in crop["padding"]], dtype=np.int64)
    after = np.asarray([item[1] for item in crop["padding"]], dtype=np.int64)
    source_start = np.maximum(start, 0)
    source_end = np.minimum(end, np.asarray(shape))
    patch_start = before
    patch_end = np.asarray(patch.shape) - after
    source_slices = tuple(
        slice(int(a), int(b)) for a, b in zip(source_start, source_end)
    )
    patch_slices = tuple(
        slice(int(a), int(b)) for a, b in zip(patch_start, patch_end)
    )
    restored = np.zeros(shape, dtype=bool)
    restored[source_slices] = patch[patch_slices]
    if not restored.any():
        raise ValueError(f"prepared component mask is empty: {entry['patch_id']}")
    return restored


def _centroid(mask):
    np = require_numpy()
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("cannot select slices from an empty mask")
    return tuple(int(v) for v in np.rint(coordinates.mean(axis=0)))


def _bbox(mask, padding: int = 0):
    np = require_numpy()
    coordinates = np.argwhere(mask)
    minimum = np.maximum(coordinates.min(axis=0) - int(padding), 0)
    maximum = np.minimum(
        coordinates.max(axis=0) + 1 + int(padding), np.asarray(mask.shape)
    )
    return tuple(slice(int(a), int(b)) for a, b in zip(minimum, maximum))


_VIEWS = (("Axial", 2), ("Coronal", 1), ("Sagittal", 0))


def _slice(array, center, axis):
    np = require_numpy()
    return np.rot90(np.take(array, int(center[axis]), axis=axis))


def _window(image, percentiles):
    np = require_numpy()
    values = np.asarray(image)
    finite = values[np.isfinite(values)]
    foreground = finite[np.abs(finite) > 1e-6]
    values = foreground if foreground.size else finite
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [float(percentiles[0]), float(percentiles[1])])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _plot_modules():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for visualization") from exc
    return plt


def _contour(ax, mask_slice, color: str, linewidth: float = 1.0):
    np = require_numpy()
    if np.any(mask_slice) and not np.all(mask_slice):
        ax.contour(
            mask_slice.astype(np.float32),
            levels=[0.5],
            colors=[color],
            linewidths=linewidth,
        )


def _overlay_mask(ax, mask_slice, color: str, alpha: float):
    np = require_numpy()
    if not np.any(mask_slice):
        return
    from matplotlib.colors import ListedColormap

    overlay = np.ma.masked_where(~mask_slice, mask_slice.astype(np.float32))
    ax.imshow(overlay, cmap=ListedColormap([color]), alpha=float(alpha), vmin=0, vmax=1)


def _save_figure(fig, path: Path, dpi: int):
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    _plot_modules().close(fig)


def _plot_mask(mask, center, spacing, roi_meta, path, cfg):
    plt = _plot_modules()
    volume_mm3 = float(mask.sum()) * float(spacing[0] * spacing[1] * spacing[2])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, axis) in zip(axes, _VIEWS):
        ax.imshow(_slice(mask, center, axis), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{name} | index {center[axis]}")
        ax.axis("off")
    fig.suptitle(
        f"Lesion mask | {int(mask.sum())} voxels | {volume_mm3:.1f} mm³ | "
        f"bbox {tuple(roi_meta['bbox_shape'])}"
    )
    _save_figure(fig, path, cfg["dpi"])


def _plot_generated(generated, mask, center, path, cfg, window):
    np = require_numpy()
    plt = _plot_modules()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, axis) in zip(axes, _VIEWS):
        image_slice = _slice(generated, center, axis)
        mask_slice = _slice(mask, center, axis)
        shown = np.ma.masked_where(~mask_slice, image_slice)
        ax.imshow(shown, cmap="gray", vmin=window[0], vmax=window[1])
        _contour(ax, mask_slice, cfg["mask_color"])
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("Generated lesion (mask exterior hidden)")
    _save_figure(fig, path, cfg["dpi"])


def _plot_full(before, after, mask, center, path, cfg, window):
    plt = _plot_modules()
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for row, (name, axis) in enumerate(_VIEWS):
        before_slice = _slice(before, center, axis)
        after_slice = _slice(after, center, axis)
        mask_slice = _slice(mask, center, axis)
        for ax, image, title in zip(
            axes[row],
            (before_slice, after_slice, after_slice),
            ("Before", "After", "After + mask"),
        ):
            ax.imshow(image, cmap="gray", vmin=window[0], vmax=window[1])
            if title.endswith("mask"):
                _overlay_mask(
                    ax, mask_slice, cfg["mask_color"], cfg["mask_alpha"]
                )
                _contour(ax, mask_slice, cfg["mask_color"])
            ax.set_title(f"{name}: {title}")
            ax.axis("off")
    fig.suptitle("Full-volume synthesis comparison")
    _save_figure(fig, path, cfg["dpi"])


def _plot_zoom(before, generated, after, mask, center, path, cfg, window):
    np = require_numpy()
    plt = _plot_modules()
    difference = masked_absolute_difference(before, after, mask)
    vmax_difference = max(float(difference.max()), 1e-6)
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    columns = ("Before", "Generated", "After", "Absolute difference")
    for row, (name, axis) in enumerate(_VIEWS):
        mask_slice = _slice(mask, center, axis)
        images = (
            _slice(before, center, axis),
            _slice(generated, center, axis),
            _slice(after, center, axis),
            _slice(difference, center, axis),
        )
        for column, (ax, image, title) in enumerate(zip(axes[row], images, columns)):
            if column == 3:
                ax.imshow(image, cmap="magma", vmin=0, vmax=vmax_difference)
            else:
                shown = (
                    np.ma.masked_where(~mask_slice, image)
                    if column == 1
                    else image
                )
                ax.imshow(shown, cmap="gray", vmin=window[0], vmax=window[1])
            _contour(ax, mask_slice, cfg["mask_color"])
            ax.set_title(f"{name}: {title}")
            ax.axis("off")
    fig.suptitle("Lesion ROI comparison")
    _save_figure(fig, path, cfg["dpi"])


def _plot_multislice(before, after, mask, path, cfg, window):
    np = require_numpy()
    plt = _plot_modules()
    indices = np.flatnonzero(mask.any(axis=(0, 1)))
    maximum = max(int(cfg["max_contact_slices"]), 1)
    if len(indices) > maximum:
        selected = np.linspace(0, len(indices) - 1, maximum).round().astype(int)
        indices = indices[selected]
    fig, axes = plt.subplots(
        2, len(indices), figsize=(max(3 * len(indices), 6), 6), squeeze=False
    )
    for column, index in enumerate(indices):
        center = (0, 0, int(index))
        mask_slice = _slice(mask, center, 2)
        for row, (volume, label) in enumerate(((before, "Before"), (after, "After"))):
            axes[row, column].imshow(
                _slice(volume, center, 2),
                cmap="gray",
                vmin=window[0],
                vmax=window[1],
            )
            _contour(axes[row, column], mask_slice, cfg["mask_color"], 0.8)
            axes[row, column].set_title(f"{label} z={index}")
            axes[row, column].axis("off")
    fig.suptitle("Axial slices across lesion extent")
    _save_figure(fig, path, cfg["dpi"])


def _plot_qc(before, generated, after, mask, qc, path, cfg):
    np = require_numpy()
    plt = _plot_modules()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = 40
    axes[0].hist(before[mask], bins=bins, alpha=0.55, label="Before")
    axes[0].hist(generated[mask], bins=bins, alpha=0.55, label="Generated")
    axes[0].hist(after[mask], bins=bins, histtype="step", linewidth=1.5, label="After")
    axes[0].set_xlabel("Intensity")
    axes[0].set_ylabel("Voxel count")
    axes[0].set_title("Lesion-region intensity distributions")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    status = "PASSED" if qc["passed"] else "FAILED"
    color = "#247a3c" if qc["passed"] else "#b3261e"
    failures = ", ".join(qc["failures"]) if qc["failures"] else "none"
    difference = masked_absolute_difference(before, after, mask)
    lines = [
        f"QC: {status}",
        f"Mask voxels: {qc['mask_voxels']}",
        f"Boundary jump (z): {qc['boundary_jump']:.4f}",
        f"Background exact: {qc['background_exact']}",
        f"Mean |difference|: {float(difference[mask].mean()):.4f}",
        f"Max |difference|: {float(difference[mask].max()):.4f}",
        f"Failures: {failures}",
    ]
    axes[1].text(
        0.04,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        color=color,
        transform=axes[1].transAxes,
        linespacing=1.7,
    )
    axes[1].set_title("Synthesis quality control")
    axes[1].axis("off")
    fig.suptitle(f"QC diagnostics — {status}", color=color)
    _save_figure(fig, path, cfg["dpi"])


def _plot_comparison(before, generated, after, mask, center, path, cfg, window):
    np = require_numpy()
    plt = _plot_modules()
    difference = masked_absolute_difference(before, after, mask)
    vmax_difference = max(float(difference.max()), 1e-6)
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    columns = ("Mask", "Before", "Generated", "After", "Difference")
    for row, (name, axis) in enumerate(_VIEWS):
        mask_slice = _slice(mask, center, axis)
        generated_slice = np.ma.masked_where(
            ~mask_slice, _slice(generated, center, axis)
        )
        images = (
            mask_slice,
            _slice(before, center, axis),
            generated_slice,
            _slice(after, center, axis),
            _slice(difference, center, axis),
        )
        for column, (ax, image, title) in enumerate(zip(axes[row], images, columns)):
            if column == 0:
                ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            elif column == 4:
                ax.imshow(image, cmap="magma", vmin=0, vmax=vmax_difference)
            else:
                ax.imshow(image, cmap="gray", vmin=window[0], vmax=window[1])
                _contour(ax, mask_slice, cfg["mask_color"], 0.8)
            ax.set_title(f"{name}: {title}")
            ax.axis("off")
    fig.suptitle("LeFusion-H fixed-mask synthesis")
    _save_figure(fig, path, cfg["dpi"])


def _render_figures(before, generated, after, mask, spacing, roi_meta, output, cfg):
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    center = _centroid(mask)
    zoom_roi = _bbox(mask, int(cfg["roi_padding"]))
    zoom_mask = mask[zoom_roi]
    zoom_center = _centroid(zoom_mask)
    zoom_before = before[zoom_roi]
    zoom_generated = generated[zoom_roi]
    zoom_after = after[zoom_roi]
    window = _window(before, cfg["intensity_percentiles"])

    paths = {
        "mask": figures / "01_mask_orthogonal.png",
        "generated": figures / "02_generated_lesion.png",
        "full_comparison": figures / "03_before_after_full.png",
        "zoom_comparison": figures / "04_before_after_zoom.png",
        "multislice": figures / "05_multislice_axial.png",
        "qc": figures / "06_intensity_qc.png",
        "comparison": figures / "comparison.png",
    }
    _plot_mask(mask, center, spacing, roi_meta, paths["mask"], cfg)
    _plot_generated(
        zoom_generated, zoom_mask, zoom_center, paths["generated"], cfg, window
    )
    _plot_full(before, after, mask, center, paths["full_comparison"], cfg, window)
    _plot_zoom(
        zoom_before,
        zoom_generated,
        zoom_after,
        zoom_mask,
        zoom_center,
        paths["zoom_comparison"],
        cfg,
        window,
    )
    _plot_multislice(
        before, after, mask, paths["multislice"], cfg, window
    )
    return paths, center, window


def _save_patch_nifti(data, ras_reference, start, path):
    np = require_numpy()
    nib = require_nibabel()
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = np.asarray(start, dtype=np.float64)
    affine = ras_reference.affine @ translation
    header = ras_reference.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(
        nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine, header),
        str(path),
    )


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
    np = require_numpy()
    image, mask, image_ras, image_source = _load_aligned_ras(image_path, mask_path)
    if prepared_entry is not None:
        if prepared_dir is None:
            raise ValueError("prepared_dir is required for a prepared component")
        mask = _mask_from_prepared_entry(image.shape, prepared_dir, prepared_entry)
    patch_shape = tuple(int(v) for v in config["data"]["patch_size"])
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
        _save_patch_nifti(generated_raw, image_ras, roi_meta["start"], patch_output)

    cfg = config["visualization"]
    paths, center, window = _render_figures(
        image,
        generated_raw if generated_raw.shape == image.shape else _embed_patch(
            image.shape, generated_raw, roi
        ),
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
        _embed_patch(image.shape, generated_raw, roi),
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
        "slice_center_ras_voxel": [int(v) for v in center],
        "roi_ras_voxel": roi_meta,
        "normalization": normalization,
        "histogram": histogram.tolist(),
        "qc": qc,
        "geometry": {
            "shape": [int(v) for v in image_source.shape],
            "spacing": [float(v) for v in image_source.header.get_zooms()[:3]],
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


def _embed_patch(shape, patch, roi):
    np = require_numpy()
    result = np.zeros(shape, dtype=np.asarray(patch).dtype)
    result[roi] = patch
    return result


def _write_case_index(output_dir: Path, records):
    csv_path = output_dir / "index.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("case_id", "qc_passed", "failures", "output_dir", "comparison"),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "case_id": record["case_id"],
                    "qc_passed": record["qc"]["passed"],
                    "failures": ";".join(record["qc"]["failures"]),
                    "output_dir": str(Path(record["outputs"]["synthetic"]).parent),
                    "comparison": record["outputs"]["figures"]["comparison"],
                }
            )

    cards = []
    for record in records:
        case = html.escape(record["case_id"])
        status = "PASS" if record["qc"]["passed"] else "FAIL"
        comparison = Path(record["outputs"]["figures"]["comparison"])
        relative = comparison.relative_to(output_dir).as_posix()
        failures = html.escape(", ".join(record["qc"]["failures"]) or "none")
        cards.append(
            f'<article><h2>{case} — {status}</h2>'
            f'<a href="{html.escape(relative)}"><img src="{html.escape(relative)}" '
            f'alt="Synthesis comparison for {case}"></a>'
            f"<p>QC failures: {failures}</p></article>"
        )
    document = (
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>LeFusion-H visualization index</title>"
        "<style>body{font-family:Arial,sans-serif;margin:2rem;max-width:1400px}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));"
        "gap:1.5rem}article{border:1px solid #bbb;padding:1rem}img{width:100%;height:auto}"
        "h1,h2{font-weight:500}</style><h1>LeFusion-H visualization index</h1><main>"
        + "".join(cards)
        + "</main></html>"
    )
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def visualize(
    config: dict[str, Any],
    *,
    image: str | Path | None = None,
    mask: str | Path | None = None,
    output_dir: str | Path | None = None,
    case_id: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run one fixed-mask case or all train and validation cases by default."""
    np = require_numpy()
    torch = require_torch()
    if bool(image) != bool(mask):
        raise ValueError("--image and --mask must be provided together")
    visualization_cfg = config["visualization"]
    if str(visualization_cfg.get("format", "png")).lower() != "png":
        raise ValueError("visualization.format currently supports only png")
    if int(visualization_cfg["dpi"]) <= 0:
        raise ValueError("visualization.dpi must be positive")
    if not 0.0 <= float(visualization_cfg["mask_alpha"]) <= 1.0:
        raise ValueError("visualization.mask_alpha must be between 0 and 1")
    percentiles = visualization_cfg["intensity_percentiles"]
    if (
        len(percentiles) != 2
        or not 0 <= float(percentiles[0]) < float(percentiles[1]) <= 100
    ):
        raise ValueError(
            "visualization.intensity_percentiles must be two increasing values in [0, 100]"
        )

    base_seed = int(config["seed"] if seed is None else seed)
    configured_output = Path(visualization_cfg["output_dir"])
    requested_output = Path(output_dir) if output_dir else configured_output
    single_case = bool(image)
    if single_case:
        resolved_case = case_id or _case_id(image)
        entries = [{"case_id": resolved_case, "image": Path(image), "mask": Path(mask)}]
        root_output = requested_output if output_dir else requested_output / resolved_case
    else:
        if case_id:
            raise ValueError("--case-id is only valid together with --image and --mask")
        prepared_dir = Path(config["data"]["prepared_dir"])
        split_path = prepared_dir / "split.json"
        manifest_path = prepared_dir / "manifest.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"prepared split not found: {split_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"prepared manifest not found: {manifest_path}")
        split = read_json(split_path)
        prepared_manifest = read_json(manifest_path)
        cases = list(
            dict.fromkeys(
                list(split.get("cases", {}).get("train", []))
                + list(split.get("cases", {}).get("val", []))
            )
        )
        if not cases:
            raise ValueError("prepared split contains no train or validation cases")
        dataset = Path(config["data"]["source_dataset"])
        channel = int(config["data"]["channel"])
        candidates_by_case = {}
        for candidate in prepared_manifest.get("entries", []):
            candidate_case = candidate["case_id"]
            current = candidates_by_case.get(candidate_case)
            if current is None or (
                int(candidate["voxels"]),
                -int(candidate["component_id"]),
            ) > (
                int(current["voxels"]),
                -int(current["component_id"]),
            ):
                candidates_by_case[candidate_case] = candidate
        missing = [item for item in cases if item not in candidates_by_case]
        if missing:
            raise ValueError(
                "train/validation cases have no valid prepared lesion component: "
                + ", ".join(missing)
            )
        entries = [
            {
                "case_id": item,
                "image": image_path(dataset, item, channel),
                "mask": label_path(dataset, item),
                "prepared_entry": candidates_by_case[item],
            }
            for item in cases
        ]
        root_output = requested_output

    for entry in entries:
        if not entry["case_id"]:
            raise ValueError("case_id must not be empty")
        if not entry["image"].is_file():
            raise FileNotFoundError(f"image not found: {entry['image']}")
        if not entry["mask"].is_file():
            raise FileNotFoundError(f"mask not found: {entry['mask']}")

    histogram_path = Path(config["data"]["prepared_dir"]) / "train_histograms.npy"
    if not histogram_path.is_file():
        raise FileNotFoundError(f"training histogram library not found: {histogram_path}")
    histogram_library = np.load(histogram_path).copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_checkpoint(
        config["synthesis"]["checkpoint"], config["model"], device
    )
    logger.info("Visualizing %d case(s) on %s", len(entries), device)

    records = []
    for index, entry in enumerate(entries):
        target = root_output if single_case else root_output / entry["case_id"]
        records.append(
            _process_case(
                config,
                model,
                checkpoint,
                device,
                image_path=entry["image"],
                mask_path=entry["mask"],
                output_dir=target,
                case_id=entry["case_id"],
                seed=base_seed + index,
                histogram_library=histogram_library,
                prepared_entry=entry.get("prepared_entry"),
                prepared_dir=(
                    Path(config["data"]["prepared_dir"])
                    if entry.get("prepared_entry") is not None
                    else None
                ),
            )
        )
    if not single_case:
        root_output.mkdir(parents=True, exist_ok=True)
        _write_case_index(root_output, records)

    return {
        "cases": len(records),
        "qc_passed": sum(bool(record["qc"]["passed"]) for record in records),
        "output_dir": str(root_output),
        "records": [record["case_id"] for record in records],
    }
