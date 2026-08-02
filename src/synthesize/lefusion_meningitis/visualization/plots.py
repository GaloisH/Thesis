from __future__ import annotations

from pathlib import Path

from ..io import require_numpy

_VIEWS = (("Axial", 2), ("Coronal", 1), ("Sagittal", 0))


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


def _centroid(mask):
    np = require_numpy()
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("cannot select slices from an empty mask")
    return tuple(int(value) for value in np.rint(coordinates.mean(axis=0)))


def _bbox(mask, padding: int = 0):
    np = require_numpy()
    coordinates = np.argwhere(mask)
    minimum = np.maximum(coordinates.min(axis=0) - int(padding), 0)
    maximum = np.minimum(
        coordinates.max(axis=0) + 1 + int(padding), np.asarray(mask.shape)
    )
    return tuple(slice(int(a), int(b)) for a, b in zip(minimum, maximum))


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
                _overlay_mask(ax, mask_slice, cfg["mask_color"], cfg["mask_alpha"])
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
                shown = np.ma.masked_where(~mask_slice, image) if column == 1 else image
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
    plt = _plot_modules()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(before[mask], bins=40, alpha=0.55, label="Before")
    axes[0].hist(generated[mask], bins=40, alpha=0.55, label="Generated")
    axes[0].hist(after[mask], bins=40, histtype="step", linewidth=1.5, label="After")
    axes[0].set_title("Lesion-region intensity distributions")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    status = "PASSED" if qc["passed"] else "FAILED"
    color = "#247a3c" if qc["passed"] else "#b3261e"
    difference = masked_absolute_difference(before, after, mask)
    failures = ", ".join(qc["failures"]) or "none"
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
        0.04, 0.95, "\n".join(lines), va="top", color=color, transform=axes[1].transAxes
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
        images = (
            mask_slice,
            _slice(before, center, axis),
            np.ma.masked_where(~mask_slice, _slice(generated, center, axis)),
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


def embed_patch(shape, patch, roi):
    np = require_numpy()
    result = np.zeros(shape, dtype=np.asarray(patch).dtype)
    result[roi] = patch
    return result


def render_figures(before, generated, after, mask, spacing, roi_meta, output, cfg):
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    center = _centroid(mask)
    zoom_roi = _bbox(mask, int(cfg["roi_padding"]))
    zoom_mask = mask[zoom_roi]
    zoom_center = _centroid(zoom_mask)
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
    _plot_generated(generated[zoom_roi], zoom_mask, zoom_center, paths["generated"], cfg, window)
    _plot_full(before, after, mask, center, paths["full_comparison"], cfg, window)
    _plot_zoom(
        before[zoom_roi],
        generated[zoom_roi],
        after[zoom_roi],
        zoom_mask,
        zoom_center,
        paths["zoom_comparison"],
        cfg,
        window,
    )
    _plot_multislice(before, after, mask, paths["multislice"], cfg, window)
    return paths, center, window

