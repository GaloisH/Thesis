"""
Compare two .nii.gz mask folders using SimpleITK.

For each case present in both folders:
    1. Verify that Size, Spacing, Direction, and Origin match exactly.
    2. Traverse every slice along all three axes (axial, coronal, sagittal).
       If a slice has more than `mismatch_threshold` mismatched voxels **and**
       its IoU is below `iou_threshold`, save a 2×3 comparison image:

       ┌──────────┬──────────┬──────────┐
       │  pred    │   gt     │  diff    │  ← mask slices (grayscale)
       ├──────────┼──────────┼──────────┤
       │ pred on  │  gt on   │ 3-color  │  ← overlay on brain MR image
       │ brain    │  brain   │ compare  │
       └──────────┴──────────┴──────────┘

Usage (CLI):
    python compare_masks.py \\
        --pred_dir datasets/dataset0610 \\
        --gt_dir datasets/修改南大数据 \\
        --images_dir datasets/nnUNet_raw/Dataset003_Meningitis/imagesTs \\
        --output_dir outputs/mask_comparison \\
        --case_ids case_000 case_001 \\
        --mismatch_threshold 10 \\
        --iou_threshold 0.95

Usage (API):
    from src.statistics.compare_masks import compare_masks
    compare_masks(
        folders=("datasets/dataset0610", "datasets/修改南大数据"),
        case_ids=["case_000"],
        images_dir="datasets/nnUNet_raw/Dataset003_Meningitis/imagesTs",
        output_dir="outputs/mask_comparison",
    )
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Windows path workaround – SimpleITK's C++ backend can't open paths with
# non-ASCII characters on Windows.  We work around this by converting to the
# 8.3 short-path form, or (as a fallback) by copying to a temp file.
# ---------------------------------------------------------------------------

_TEMP_COPIES: list[Path] = []  # track temp files for cleanup


def _readable_path(path: Path) -> Path:
    """Return a path that SimpleITK can actually open.

    On Windows with non-ASCII paths we copy the file to a temp location with
    an ASCII-safe name, because SimpleITK's C++ backend uses narrow-string
    APIs that can't handle Unicode paths, and Windows 8.3 short names strip
    the double extension (``.nii.gz``) that SimpleITK relies on for format
    detection.
    """
    try:
        str(path).encode("ascii")
        return path
    except UnicodeEncodeError:
        pass

    # Temp-copy fallback for non-ASCII paths
    tmp = Path(tempfile.gettempdir()) / f"_sitk_{path.name}"
    if not tmp.exists():
        shutil.copy2(path, tmp)
    _TEMP_COPIES.append(tmp)
    return tmp


def _cleanup_temp_copies() -> None:
    """Remove any temp copies created during this run."""
    for p in _TEMP_COPIES:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    _TEMP_COPIES.clear()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("compare_masks")
logger.setLevel(logging.DEBUG)

# Console handler
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.DEBUG)
_ch.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(_ch)

# File handler (attached once output_dir is known)
_fh: Optional[logging.FileHandler] = None


def _attach_file_log(output_dir: Path) -> None:
    """Attach a file handler so logs are also written to *output_dir/compare.log*."""
    global _fh
    if _fh is not None:
        logger.removeHandler(_fh)
    output_dir.mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(output_dir / "compare.log", encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_fh)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_cases(pred_dir: Path, gt_dir: Path, case_ids: Optional[Sequence[str]]) -> list[str]:
    """Return sorted case IDs that exist in **both** directories.

    If *case_ids* is ``None``, the intersection of all ``case_*.nii.gz`` stems
    in the two folders is used.
    """
    pred_stems = {p.stem.removesuffix(".nii") for p in pred_dir.glob("case_*.nii.gz")}
    gt_stems = {p.stem.removesuffix(".nii") for p in gt_dir.glob("case_*.nii.gz")}

    if case_ids is None:
        common = sorted(pred_stems & gt_stems)
    else:
        common = sorted(set(case_ids) & pred_stems & gt_stems)

    missing_pred = (set(case_ids or []) if case_ids is not None else set()) - pred_stems
    missing_gt = (set(case_ids or []) if case_ids is not None else set()) - gt_stems
    if missing_pred:
        logger.warning("Cases not found in pred_dir (%s): %s", pred_dir, sorted(missing_pred))
    if missing_gt:
        logger.warning("Cases not found in gt_dir   (%s): %s", gt_dir, sorted(missing_gt))

    logger.info("Discovered %d common case(s) to process.", len(common))
    return common


def _check_metadata(pred: sitk.Image, gt: sitk.Image, case_id: str) -> None:
    """Raise ``ValueError`` if the two images have mismatched geometry.

    Float fields (Spacing, Direction, Origin) are compared with
    ``np.allclose`` at ``rtol=1e-5`` to tolerate harmless floating-point
    round-off between different NIfTI writers.
    """
    # Integer fields: exact match
    if pred.GetSize() != gt.GetSize():
        msg = f"[{case_id}] Size mismatch!  Pred: {pred.GetSize()}  |  GT: {gt.GetSize()}"
        logger.error(msg)
        raise ValueError(msg)

    # Float fields: allclose
    float_fields = {
        "Spacing": (pred.GetSpacing(), gt.GetSpacing()),
        "Direction": (pred.GetDirection(), gt.GetDirection()),
        "Origin": (pred.GetOrigin(), gt.GetOrigin()),
    }
    for name, (v_pred, v_gt) in float_fields.items():
        if not np.allclose(v_pred, v_gt, rtol=1e-5):
            msg = f"[{case_id}] {name} mismatch!  Pred: {v_pred}  |  GT: {v_gt}"
            logger.error(msg)
            raise ValueError(msg)

    logger.info(
        "[%s] Metadata OK – Size=%s Spacing=%s",
        case_id, pred.GetSize(), pred.GetSpacing(),
    )


def _normalize_brain(slice_2d: np.ndarray) -> np.ndarray:
    """Clip a brain-MR slice to [p1, p99] and rescale to [0, 1] float."""
    vmin, vmax = np.percentile(slice_2d[slice_2d > 0] if (slice_2d > 0).any() else slice_2d, [1, 99])
    if vmax <= vmin:
        vmax = vmin + 1
    return np.clip((slice_2d.astype(np.float32) - vmin) / (vmax - vmin), 0, 1)


def _draw_comparison_slice(
    pred_slice: np.ndarray,
    gt_slice: np.ndarray,
    brain_slice: Optional[np.ndarray],
    dim_name: str,
    slice_idx: int,
) -> plt.Figure:
    """Return a matplotlib Figure with a 2×3 panel layout.

    **Top row** (grayscale masks):
        pred | gt | difference

    **Bottom row** (overlays on brain MR, only when *brain_slice* is provided):
        pred-on-brain (red) | gt-on-brain (red) | 3-color compare
        - Green  = both pred & gt agree
        - Red    = pred only (false positive)
        - Blue   = gt only  (false negative)
    """
    diff = (pred_slice != gt_slice).astype(np.uint8)
    has_brain = brain_slice is not None

    n_rows = 2 if has_brain else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)  # make it 2-D for uniform indexing

    # -- Row 0: mask slices (grayscale) --
    titles_top = [
        f"Prediction ({dim_name} #{slice_idx})",
        f"Ground Truth ({dim_name} #{slice_idx})",
        f"Difference ({dim_name} #{slice_idx})",
    ]
    arrays_top = [pred_slice, gt_slice, diff]

    for ax, arr, title in zip(axes[0], arrays_top, titles_top):
        im = ax.imshow(arr.T, cmap="gray", origin="lower", interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # -- Row 1: brain overlay panels --
    if has_brain:
        brain_norm = _normalize_brain(brain_slice)

        # --- Panel 4: brain + pred mask (red overlay) ---
        ax4 = axes[1, 0]
        ax4.imshow(brain_norm.T, cmap="gray", origin="lower", interpolation="nearest")
        mask_pred = (pred_slice > 0).astype(np.uint8)
        if mask_pred.any():
            ax4.imshow(
                np.ma.masked_where(mask_pred.T == 0, mask_pred.T),
                cmap="Reds", alpha=0.55, origin="lower", interpolation="nearest",
            )
        ax4.set_title(f"Prediction Overlay ({dim_name} #{slice_idx})", fontsize=9)
        ax4.axis("off")

        # --- Panel 5: brain + gt mask (red overlay) ---
        ax5 = axes[1, 1]
        ax5.imshow(brain_norm.T, cmap="gray", origin="lower", interpolation="nearest")
        mask_gt = (gt_slice > 0).astype(np.uint8)
        if mask_gt.any():
            ax5.imshow(
                np.ma.masked_where(mask_gt.T == 0, mask_gt.T),
                cmap="Reds", alpha=0.55, origin="lower", interpolation="nearest",
            )
        ax5.set_title(f"Ground Truth Overlay ({dim_name} #{slice_idx})", fontsize=9)
        ax5.axis("off")

        # --- Panel 6: 3-color comparison ---
        # Green  = both present  (intersection)
        # Red    = pred only      (false positive)
        # Blue   = gt only        (false negative)
        ax6 = axes[1, 2]
        ax6.imshow(brain_norm.T, cmap="gray", origin="lower", interpolation="nearest")

        both = np.logical_and(pred_slice > 0, gt_slice > 0)
        pred_only = np.logical_and(pred_slice > 0, gt_slice == 0)
        gt_only = np.logical_and(pred_slice == 0, gt_slice > 0)

        # Build RGB overlay: (R, G, B) per voxel
        rgb = np.zeros((*pred_slice.shape, 3), dtype=np.float32)
        rgb[both, :] = [0, 1, 0]         # Green
        rgb[pred_only, :] = [1, 0, 0]    # Red
        rgb[gt_only, :] = [0, 0, 1]      # Blue

        # Transpose to (H, W, 3) for imshow
        rgb_display = np.transpose(rgb, (1, 0, 2))
        # Alpha composite: only show color where there's signal
        alpha = (rgb_display.sum(axis=-1) > 0).astype(np.float32) * 0.65

        ax6.imshow(rgb_display, alpha=alpha, origin="lower", interpolation="nearest")
        ax6.set_title(f"3-Color Compare ({dim_name} #{slice_idx})", fontsize=9)
        ax6.axis("off")

        # Legend for panel 6
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="green", alpha=0.65, label="Both"),
            Patch(facecolor="red", alpha=0.65, label="Pred only (FP)"),
            Patch(facecolor="blue", alpha=0.65, label="GT only (FN)"),
        ]
        ax6.legend(handles=legend_elements, loc="lower right", fontsize=7,
                   framealpha=0.8, ncol=1)

    fig.tight_layout()
    return fig


def _dimension_name(axis: int) -> str:
    """Map axis index (in SimpleITK numpy order: z=0, y=1, x=2) to a readable label."""
    return {0: "axial", 1: "coronal", 2: "sagittal"}[axis]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def compare_masks(
    folders: tuple[str, str] | tuple[Path, Path],
    case_ids: Optional[Sequence[str]] = None,
    output_dir: str | Path = "outputs/mask_comparison",
    mismatch_threshold: int = 0,
    iou_threshold: float = 0.95,
    images_dir: Optional[str | Path] = None,
    image_channel: str = "_0001",
) -> None:
    """Main entry point – compare two folders of .nii.gz masks.

    Parameters
    ----------
    folders:
        ``(prediction_dir, ground_truth_dir)``.
    case_ids:
        Case IDs to process (e.g. ``["case_000", "case_001"]``).
        ``None`` means process every case found in both folders.
    output_dir:
        Root output directory.  A timestamped sub-directory is created inside
        it, and each case gets its own sub-directory within that.
    mismatch_threshold:
        Minimum number of mismatched voxels on a slice to trigger saving.
    iou_threshold:
        Maximum IoU on a slice to trigger saving (lower = more strict).
    images_dir:
        Optional path to the original brain MR images (e.g.
        ``datasets/nnUNet_raw/Dataset003_Meningitis/imagesTs``).
        Files are expected as ``{case_id}{image_channel}.nii.gz``.  When
        provided, the bottom row of the output figure overlays the masks on
        the brain image.
    image_channel:
        Channel suffix to select from *images_dir*.  Default ``"_0001"``
        (typically T1ce).
    """
    pred_dir = Path(folders[0])
    gt_dir = Path(folders[1])
    img_dir = Path(images_dir) if images_dir is not None else None

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")
    if img_dir is not None and not img_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")

    # --- Setup output tree ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / ts
    _attach_file_log(run_dir)

    logger.info("=" * 60)
    logger.info("Mask Comparison Run")
    logger.info("  Prediction dir  : %s", pred_dir)
    logger.info("  Ground-truth dir: %s", gt_dir)
    logger.info("  Brain images dir: %s", img_dir or "(none)")
    logger.info("  Image channel   : %s", image_channel if img_dir else "(n/a)")
    logger.info("  Output dir      : %s", run_dir)
    logger.info("  Mismatch thresh : %d voxels", mismatch_threshold)
    logger.info("  IoU threshold   : %.4f", iou_threshold)
    logger.info("=" * 60)

    # --- Discover cases ---
    cases = _discover_cases(pred_dir, gt_dir, case_ids)
    if not cases:
        logger.warning("No matching cases found – exiting.")
        return

    # --- Process each case ---
    summary: dict[str, dict] = {}

    for case_id in cases:
        case_out = run_dir / case_id
        case_out.mkdir(parents=True, exist_ok=True)

        pred_path = pred_dir / f"{case_id}.nii.gz"
        gt_path = gt_dir / f"{case_id}.nii.gz"

        logger.info("--- Processing %s ---", case_id)

        # Read masks
        pred_img = sitk.ReadImage(str(_readable_path(pred_path)))
        gt_img = sitk.ReadImage(str(_readable_path(gt_path)))

        # 1. Metadata check (masks)
        try:
            _check_metadata(pred_img, gt_img, case_id)
        except ValueError:
            logger.exception("Metadata mismatch for %s – skipping.", case_id)
            continue

        # 2. Convert to numpy (z, y, x)
        pred_arr = sitk.GetArrayFromImage(pred_img).astype(np.int16)
        gt_arr = sitk.GetArrayFromImage(gt_img).astype(np.int16)

        if pred_arr.shape != gt_arr.shape:
            logger.error(
                "[%s] Array shape mismatch after read: pred=%s gt=%s",
                case_id, pred_arr.shape, gt_arr.shape,
            )
            continue

        # 3. Load brain image if available
        brain_arr: Optional[np.ndarray] = None
        if img_dir is not None:
            brain_path = img_dir / f"{case_id}{image_channel}.nii.gz"
            if brain_path.is_file():
                brain_img = sitk.ReadImage(str(_readable_path(brain_path)))
                brain_arr = sitk.GetArrayFromImage(brain_img).astype(np.float32)
                # Verify brain-vs-mask geometry
                if brain_arr.shape != pred_arr.shape:
                    logger.warning(
                        "[%s] Brain image shape %s ≠ mask shape %s – overlay disabled for this case.",
                        case_id, brain_arr.shape, pred_arr.shape,
                    )
                    brain_arr = None
                else:
                    logger.info("[%s] Brain image loaded – overlay enabled.", case_id)
            else:
                logger.warning(
                    "[%s] Brain image not found at %s – overlay disabled for this case.",
                    case_id, brain_path,
                )

        # 4. Slice-by-slice comparison along all 3 axes
        n_dims = pred_arr.ndim  # should be 3
        total_slices = 0
        saved_slices = 0
        case_summary: dict[str, int] = {}

        for axis in range(n_dims):
            dim_name = _dimension_name(axis)
            n_slices = pred_arr.shape[axis]
            axis_saved = 0

            for idx in range(n_slices):
                # Extract 2-D mask slices
                if axis == 0:
                    p_slice = pred_arr[idx, :, :]
                    g_slice = gt_arr[idx, :, :]
                    b_slice = brain_arr[idx, :, :] if brain_arr is not None else None
                elif axis == 1:
                    p_slice = pred_arr[:, idx, :]
                    g_slice = gt_arr[:, idx, :]
                    b_slice = brain_arr[:, idx, :] if brain_arr is not None else None
                else:  # axis == 2
                    p_slice = pred_arr[:, :, idx]
                    g_slice = gt_arr[:, :, idx]
                    b_slice = brain_arr[:, :, idx] if brain_arr is not None else None

                # Compute metrics
                intersection = np.logical_and(p_slice > 0, g_slice > 0).sum()
                union = np.logical_or(p_slice > 0, g_slice > 0).sum()
                iou = intersection / union if union > 0 else 1.0
                mismatched = int((p_slice != g_slice).sum())

                # Decision: save if too many mismatches AND IoU is too low
                if mismatched > mismatch_threshold and iou < iou_threshold:
                    fig = _draw_comparison_slice(p_slice, g_slice, b_slice, dim_name, idx)
                    fname = f"layer_{dim_name}_{idx:04d}.png"
                    fig.savefig(case_out / fname, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    axis_saved += 1
                    logger.debug(
                        "[%s] Saved %s #%04d  |  mismatched=%d  IoU=%.4f",
                        case_id, dim_name, idx, mismatched, iou,
                    )

            case_summary[dim_name] = axis_saved
            saved_slices += axis_saved
            total_slices += n_slices

        summary[case_id] = {
            **case_summary,
            "total_slices": total_slices,
            "saved_slices": saved_slices,
        }

        logger.info(
            "[%s] Done – %d/%d slices saved (%d axial, %d coronal, %d sagittal)",
            case_id,
            saved_slices,
            total_slices,
            case_summary["axial"],
            case_summary["coronal"],
            case_summary["sagittal"],
        )

    # --- Final report ---
    logger.info("=" * 60)
    logger.info("Summary")
    for cid, s in summary.items():
        logger.info(
            "  %s: saved %d/%d slices (A=%d C=%d S=%d)",
            cid,
            s["saved_slices"],
            s["total_slices"],
            s["axial"],
            s["coronal"],
            s["sagittal"],
        )
    logger.info("Output written to: %s", run_dir)
    logger.info("=" * 60)
    _cleanup_temp_copies()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two .nii.gz mask folders slice-by-slice.",
    )
    parser.add_argument(
        "--pred_dir", required=True,
        help="Path to the prediction folder.",
    )
    parser.add_argument(
        "--gt_dir", required=True,
        help="Path to the ground-truth folder.",
    )
    parser.add_argument(
        "--output_dir", default="outputs/mask_comparison",
        help="Root output directory (default: outputs/mask_comparison).",
    )
    parser.add_argument(
        "--case_ids", nargs="*", default=None,
        help="Case IDs to process (space-separated). Omit to process all common cases.",
    )
    parser.add_argument(
        "--mismatch_threshold", type=int, default=0,
        help="Minimum mismatched voxels on a slice to trigger saving (default: 0).",
    )
    parser.add_argument(
        "--iou_threshold", type=float, default=0.95,
        help="Maximum IoU on a slice to trigger saving (default: 0.95).",
    )
    parser.add_argument(
        "--images_dir", default=None,
        help="Optional path to original brain MR images for overlay.",
    )
    parser.add_argument(
        "--image_channel", default="_0001",
        help="Channel suffix for brain images (default: _0001 = T1ce).",
    )

    args = parser.parse_args()
    compare_masks(
        folders=(args.pred_dir, args.gt_dir),
        case_ids=args.case_ids,
        output_dir=args.output_dir,
        mismatch_threshold=args.mismatch_threshold,
        iou_threshold=args.iou_threshold,
        images_dir=args.images_dir,
        image_channel=args.image_channel,
    )


# ---------------------------------------------------------------------------
# Test harness (invoked via `python compare_masks.py`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -- Use the exact folders requested for testing --
    PRED_DIR = r"datasets\dataset0610"
    GT_DIR = r"datasets\修改南大数据"
    IMAGES_DIR = r"datasets\nnUNet_raw\Dataset003_Meningitis\imagesTs"
    OUTPUT_DIR = r"outputs\mask_comparison"

    logger.info("Running built-in test with:")
    logger.info("  pred_dir   = %s", PRED_DIR)
    logger.info("  gt_dir     = %s", GT_DIR)
    logger.info("  images_dir = %s", IMAGES_DIR)

    # Resolve relative to this project root
    project_root = Path(__file__).resolve().parent.parent.parent

    compare_masks(
        folders=(project_root / PRED_DIR, project_root / GT_DIR),
        case_ids=["case_000"],
        output_dir=project_root / OUTPUT_DIR,
        mismatch_threshold=0,
        iou_threshold=0.95,
        images_dir=project_root / IMAGES_DIR,
        image_channel="_0001",
    )
