#!/usr/bin/env python3
"""
Compare ensemble model (nnUNet + SwinUNETR weighted fusion) with individual
models on the training set. Generates per-class Dice score statistics and 2D
slice comparison in a single combined figure.

Usage:
    cd src/inference
    python compare_models.py \
        --images_dir ../../datasets/nnUNet_raw/Dataset101_Meningioma/imagesTr \
        --labels_dir ../../datasets/nnUNet_raw/Dataset101_Meningioma/labelsTr \
        --nnunet_model_dir ../../datasets/nnUNet_results/Dataset101_Meningioma/nnUNetTrainer__nnUNetPlans__2d \
        --swin_checkpoint /path/to/best_model.pth \
        --plan_path ../../datasets/nnUNet_preprocessed/Dataset101_Meningioma/nnUNetPlans.json \
        --output_dir ../../outputs/model_comparison \
        --device cuda
"""

import os
import sys
import glob
import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Allow importing from sibling / parent modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from segmentation.plan2transform import _parse_plan
from ensemble_predict import resample_prob

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from monai.transforms import (
    Compose, LoadImaged, Orientationd, Spacingd,
    NormalizeIntensityd, ToTensord,
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

warnings.filterwarnings("ignore")

# ── Constants ───────────────────────────────────────────────────────────
TUMOR_CLASSES = [1, 2, 3]
CLASS_NAMES = ["Necrotic", "Edema", "Enhancing"]
MODEL_NAMES = ["nnUNet", "SwinUNETR", "Ensemble"]

MODEL_COLORS = {
    "nnUNet":    "#3498db",
    "SwinUNETR": "#e67e22",
    "Ensemble":  "#e74c3c",
}
CONTOUR_COLORS = {
    "GT":        "#2ecc71",
    "nnUNet":    "#3498db",
    "SwinUNETR": "#e67e22",
    "Ensemble":  "#e74c3c",
}
CLASS_CONTOUR_COLORS = {1: "#ff9999", 2: "#99ccff", 3: "#ffcc66"}


# ── Small utilities (kept local to avoid heavy transitive imports) ──────

def dice_score(pred: np.ndarray, gt: np.ndarray, class_idx: int) -> float:
    """Compute Dice coefficient for a single class."""
    p = (pred == class_idx)
    g = (gt == class_idx)
    intersection = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return np.nan
    return 2.0 * intersection / denom


def pick_center(vol: np.ndarray) -> tuple:
    """Return (z, y, x) at the median of non-zero voxels."""
    nz = np.argwhere(vol > 0)
    if nz.size == 0:
        return tuple(np.array(vol.shape) // 2)
    return tuple(np.median(nz, axis=0).astype(int))


def _save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _resample_seg_to_shape(seg: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Nearest-neighbour resample a (D,H,W) integer label map to target_shape."""
    if seg.shape == target_shape:
        return seg
    t = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).float()
    return F.interpolate(t, size=target_shape, mode="nearest").squeeze().numpy().astype(np.uint8)


def _mean_dice(dice_list):
    """Mean of non-NaN dice values."""
    valid = [v for v in dice_list if not np.isnan(v)]
    return np.mean(valid) if valid else np.nan


# ── Inference: nnUNet ───────────────────────────────────────────────────

def run_nnunet_batch(image_paths_list, output_dir, model_dir, fold=0, device="cuda"):
    """Run nnUNet inference on a list of cases, caching results to disk.

    Returns:
        dict: case_id -> softmax probability array (4, D, H, W)
    """
    os.makedirs(output_dir, exist_ok=True)
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        device=torch.device(device), verbose=False,
    )
    predictor.initialize_from_trained_model_folder(model_dir, use_folds=(fold,))

    results = {}
    for i, image_paths in enumerate(image_paths_list):
        stem = Path(image_paths[0]).name.replace(".nii.gz", "").replace(".nii", "")
        case_id = stem.replace("_0000", "")
        out_path = os.path.join(output_dir, f"{stem}_softmax_prob.npy")

        if os.path.exists(out_path):
            print(f"[{i+1}/{len(image_paths_list)}] nnUNet: {case_id} (cached)")
            results[case_id] = np.load(out_path).astype(np.float32)
            continue

        print(f"[{i+1}/{len(image_paths_list)}] nnUNet: {case_id}")

        img = nib.load(image_paths[0])
        data = np.stack([nib.load(p).get_fdata(dtype=np.float32) for p in image_paths])
        props = {"spacing": img.header.get_zooms()[:3]}

        _, prob = predictor.predict_single_npy_array(
            input_image=data, image_properties=props,
            segmentation_previous_stage=None, output_file_truncated=None,
            save_or_return_probabilities=True,
        )
        softmax_prob = torch.softmax(torch.tensor(prob), dim=0).numpy().astype(np.float32)
        np.save(out_path, softmax_prob)
        results[case_id] = softmax_prob

    return results


# ── Inference: SwinUNETR ────────────────────────────────────────────────

def _build_swin_transforms(plan_path):
    """Build MONAI inference transforms from nnUNetPlans.json."""
    p = _parse_plan(plan_path)
    transforms = Compose([
        LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=p["spacing"], mode=(p["interp_image"],)),
        NormalizeIntensityd(keys=["image"], nonzero=p["use_mask_for_norm"], channel_wise=True),
        ToTensord(keys=["image"]),
    ])
    return transforms, p["patch_size"]


def run_swin_batch(image_paths_list, output_dir, checkpoint_path, plan_path, device="cuda"):
    """Run SwinUNETR inference on a list of cases, caching results to disk.

    Returns:
        dict: case_id -> softmax probability array (4, D, H, W)
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    infer_transforms, roi_size = _build_swin_transforms(plan_path)

    data_dicts = []
    for paths in image_paths_list:
        case_id = Path(paths[0]).name.replace("_0000.nii.gz", "")
        data_dicts.append({"image": list(paths), "case_id": case_id})

    model = SwinUNETR(
        in_channels=4, out_channels=4, feature_size=48,
        use_checkpoint=True, spatial_dims=3,
        depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
        window_size=(7, 7, 7), norm_name="instance",
        drop_rate=0.2, attn_drop_rate=0.0, dropout_path_rate=0.2,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dataset = Dataset(data=data_dicts, transform=infer_transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    results = {}
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            case_id = batch_data["case_id"][0]
            out_path = os.path.join(output_dir, f"{case_id}_prob.nii.gz")

            if os.path.exists(out_path):
                print(f"[{i+1}/{len(data_dicts)}] SwinUNETR: {case_id} (cached)")
                prob_nib = nib.load(out_path)
                prob = prob_nib.get_fdata().astype(np.float32)
                if prob.ndim == 4 and prob.shape[-1] == 4:
                    prob = np.transpose(prob, (3, 0, 1, 2))
                results[case_id] = prob
                continue

            print(f"[{i+1}/{len(data_dicts)}] SwinUNETR: {case_id}")

            inputs = batch_data["image"].to(device)
            outputs = sliding_window_inference(
                inputs=inputs, roi_size=roi_size, sw_batch_size=1,
                predictor=model, overlap=0.5,
            )
            probs = torch.softmax(outputs, dim=1)  # (1, 4, D, H, W)
            prob_np = probs[0].cpu().numpy().astype(np.float32)

            ref_nib = nib.load(image_paths_list[i][0])
            prob_save = np.transpose(prob_np, (1, 2, 3, 0))
            nib.save(nib.Nifti1Image(prob_save, ref_nib.affine), out_path)

            results[case_id] = prob_np

    return results


# ── Ensemble fusion ─────────────────────────────────────────────────────

def compute_ensemble_probs(nnunet_probs, swin_probs, ref_shape, w_nnunet=0.7, w_swin=0.3):
    """Weighted fusion of nnUNet and SwinUNETR softmax probabilities."""
    p_nnunet = torch.from_numpy(nnunet_probs.astype(np.float32)).unsqueeze(0)
    p_swin = torch.from_numpy(swin_probs.astype(np.float32)).unsqueeze(0)
    p_nnunet = resample_prob(p_nnunet, ref_shape)
    p_swin = resample_prob(p_swin, ref_shape)
    ensemble = w_nnunet * p_nnunet + w_swin * p_swin
    return ensemble.squeeze(0).numpy()


# ── Dice computation ────────────────────────────────────────────────────

def compute_all_dice(nnunet_probs, swin_probs, ensemble_probs, gt_path, ref_shape):
    """Compute per-class Dice for all three models against ground truth."""
    gt = nib.load(gt_path).get_fdata().astype(np.uint8)

    seg_nnunet = _resample_seg_to_shape(np.argmax(nnunet_probs, axis=0).astype(np.uint8), gt.shape)
    seg_swin   = _resample_seg_to_shape(np.argmax(swin_probs, axis=0).astype(np.uint8), gt.shape)
    seg_ens    = _resample_seg_to_shape(np.argmax(ensemble_probs, axis=0).astype(np.uint8), gt.shape)

    return {
        "nnUNet":    [dice_score(seg_nnunet, gt, c) for c in TUMOR_CLASSES],
        "SwinUNETR": [dice_score(seg_swin, gt, c) for c in TUMOR_CLASSES],
        "Ensemble":  [dice_score(seg_ens, gt, c) for c in TUMOR_CLASSES],
    }


# ── Visualization: shared slice grid drawer ─────────────────────────────

def _draw_slice_grid(axes, t1ce, segs, col_labels, center, orient_names):
    """Render a triplanar 2D slice comparison into a 2D array of axes.

    Args:
        axes: 2D np.array of matplotlib Axes (shape: 3 rows × 5 cols)
        t1ce: 3D T1ce volume
        segs: dict column_label -> 3D segmentation volume
        col_labels: list of 5 column labels
        center: (z, y, x) tuple from pick_center
        orient_names: list of 3 orientation names
    """
    z, y, x = int(center[0]), int(center[1]), int(center[2])
    slice_funcs = [
        ("Axial",    lambda vol: np.rot90(vol[z])),
        ("Coronal",  lambda vol: np.rot90(vol[:, y])),
        ("Sagittal", lambda vol: np.rot90(vol[..., x])),
    ]

    for row_idx, (orient_name, slice_fn) in enumerate(slice_funcs):
        for col_idx, col_label in enumerate(col_labels):
            ax = axes[row_idx, col_idx]

            if col_label == "T1ce":
                ax.imshow(slice_fn(t1ce), cmap="gray")
            else:
                ax.imshow(slice_fn(t1ce), cmap="gray")
                seg_slice = slice_fn(segs[col_label])

                if col_label == "GT":
                    mask = (seg_slice > 0).astype(np.float32)
                    if mask.any():
                        ax.contour(mask, levels=[0.5],
                                   colors=CONTOUR_COLORS["GT"], linewidths=1.2)
                else:
                    for cls_idx in TUMOR_CLASSES:
                        cls_mask = (seg_slice == cls_idx).astype(np.float32)
                        if cls_mask.any():
                            ax.contour(cls_mask, levels=[0.5],
                                       colors=CLASS_CONTOUR_COLORS[cls_idx],
                                       linewidths=0.8)

            if row_idx == 0:
                color = CONTOUR_COLORS.get(col_label, "black")
                ax.set_title(col_label, fontsize=11, fontweight="bold", color=color)
            if col_idx == 0:
                ax.set_ylabel(orient_names[row_idx] if orient_names else orient_name,
                              fontsize=10, fontweight="bold")
            ax.axis("off")


# ── Visualization: figure generators ────────────────────────────────────

def generate_summary_figure(dice_rows, slice_data, output_path):
    """Main combined figure: box plots (left) + 2D slice grid (right)."""
    # Aggregate dice by model and class for box plots
    dice_by_model = {m: {c: [] for c in TUMOR_CLASSES} for m in MODEL_NAMES}
    for row in dice_rows:
        for m in MODEL_NAMES:
            for ci, c in enumerate(TUMOR_CLASSES):
                val = row.get(f"{m}_d{ci+1}", np.nan)
                if not np.isnan(val):
                    dice_by_model[m][c].append(val)

    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.6], hspace=0.35)

    # Top: Box plots (4 panels: 3 classes + mean)
    gs_top = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], wspace=0.3)
    plot_groups = [(f"{cname}\n(Dice)", cidx) for cname, cidx in zip(CLASS_NAMES, TUMOR_CLASSES)] + [("Mean", "mean")]

    for ci, (clabel, ckey) in enumerate(plot_groups):
        ax = fig.add_subplot(gs_top[0, ci])

        if ckey == "mean":
            plot_data = []
            for m in MODEL_NAMES:
                means = []
                for row in dice_rows:
                    vals = [row.get(f"{m}_d{i+1}", np.nan) for i in range(3)]
                    valid = [v for v in vals if not np.isnan(v)]
                    if valid:
                        means.append(np.mean(valid))
                plot_data.append(means)
        else:
            plot_data = [dice_by_model[m][ckey] for m in MODEL_NAMES]

        bp = ax.boxplot(plot_data, patch_artist=True, widths=0.5,
                        medianprops={"color": "black", "linewidth": 1.5})
        for patch, m in zip(bp["boxes"], MODEL_NAMES):
            patch.set_facecolor(MODEL_COLORS[m])
            patch.set_alpha(0.75)

        ax.set_xticklabels(MODEL_NAMES, rotation=15, fontsize=8)
        ax.set_ylabel("Dice Score", fontsize=10)
        ax.set_title(clabel, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        for pi, (patch_data, m) in enumerate(zip(plot_data, MODEL_NAMES)):
            if patch_data:
                mu = np.nanmean(patch_data)
                ax.annotate(f"{mu:.3f}", xy=(pi + 1, mu),
                            xytext=(pi + 1, min(mu + 0.08, 0.98)),
                            ha="center", fontsize=7, fontweight="bold",
                            color=MODEL_COLORS[m])

    # Bottom: 2D slice grid (3 rows × 5 cols)
    gs_bot = GridSpecFromSubplotSpec(3, 5, subplot_spec=gs[1],
                                     wspace=0.05, hspace=0.08)
    axes_bot = np.empty((3, 5), dtype=object)
    for r in range(3):
        for c in range(5):
            axes_bot[r, c] = fig.add_subplot(gs_bot[r, c])

    segs = {
        "GT": slice_data["gt_vol"], "nnUNet": slice_data["seg_nnunet"],
        "SwinUNETR": slice_data["seg_swin"], "Ensemble": slice_data["seg_ens"],
    }
    col_labels = ["T1ce", "GT", "nnUNet", "SwinUNETR", "Ensemble"]
    center = pick_center(slice_data["gt_vol"].astype(np.float32))

    _draw_slice_grid(axes_bot, slice_data["t1ce_vol"], segs, col_labels, center,
                     orient_names=["Axial", "Coronal", "Sagittal"])

    # Global title
    case_id = slice_data["case_id"]
    d_means = {m: _mean_dice([slice_data[f"dice_{m.lower()}"][i] for i in range(3)])
               for m in ["nnUNet", "SwinUNETR", "Ensemble"]}
    n_cases = len(dice_rows)
    suptitle = (
        f"Model Comparison on Training Set (n={n_cases})   |   "
        f"Shown: {case_id} (median Ensemble Dice)   |   "
        f"Mean Dice — nnUNet: {d_means['nnUNet']:.3f}  |  "
        f"SwinUNETR: {d_means['SwinUNETR']:.3f}  |  Ensemble: {d_means['Ensemble']:.3f}"
    )
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.98)
    _save_plot(fig, output_path)
    print(f"Saved: {output_path}")


def generate_case_figure(slice_data, output_path, tag=""):
    """Triplanar comparison figure for a single case (best/worst)."""
    segs = {
        "GT": slice_data["gt_vol"], "nnUNet": slice_data["seg_nnunet"],
        "SwinUNETR": slice_data["seg_swin"], "Ensemble": slice_data["seg_ens"],
    }
    col_labels = ["T1ce", "GT", "nnUNet", "SwinUNETR", "Ensemble"]
    center = pick_center(slice_data["gt_vol"].astype(np.float32))

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    _draw_slice_grid(axes, slice_data["t1ce_vol"], segs, col_labels, center,
                     orient_names=["Axial", "Coronal", "Sagittal"])

    case_id = slice_data["case_id"]
    d_means = {m: _mean_dice([slice_data[f"dice_{m.lower()}"][i] for i in range(3)])
               for m in ["nnUNet", "SwinUNETR", "Ensemble"]}
    title = (f"{tag} Case: {case_id}   |   "
             f"nnUNet: {d_means['nnUNet']:.3f}  |  "
             f"SwinUNETR: {d_means['SwinUNETR']:.3f}  |  "
             f"Ensemble: {d_means['Ensemble']:.3f}")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    _save_plot(fig, output_path)
    print(f"Saved: {output_path}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare ensemble model vs individual models on training set"
    )
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--labels_dir", type=str, required=True)
    parser.add_argument("--nnunet_model_dir", type=str, required=True)
    parser.add_argument("--swin_checkpoint", type=str, required=True)
    parser.add_argument("--plan_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./comparison_results")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--w_nnunet", type=float, default=0.7)
    parser.add_argument("--w_swin", type=float, default=0.3)
    parser.add_argument("--skip_inference", action="store_true")
    args = parser.parse_args()

    # Setup output directories
    nnunet_out = os.path.join(args.output_dir, "nnunet_preds")
    swin_out = os.path.join(args.output_dir, "swin_preds")
    ens_out = os.path.join(args.output_dir, "ensemble_preds")
    os.makedirs(nnunet_out, exist_ok=True)
    os.makedirs(swin_out, exist_ok=True)
    os.makedirs(ens_out, exist_ok=True)

    # Gather cases
    t1_images = sorted(glob.glob(os.path.join(args.images_dir, "*_0000.nii.gz")))
    if not t1_images:
        print(f"ERROR: No images found in {args.images_dir}")
        return

    cases = []
    for t1_path in t1_images:
        case_id = Path(t1_path).name.replace("_0000.nii.gz", "")
        base = t1_path.replace("_0000.nii.gz", "")
        paths = [f"{base}_{mod}.nii.gz" for mod in ["0000", "0001", "0002", "0003"]]
        label_path = os.path.join(args.labels_dir, f"{case_id}.nii.gz")
        if os.path.exists(label_path):
            cases.append({"case_id": case_id, "image_paths": paths, "label_path": label_path})

    print(f"Found {len(cases)} valid cases with labels")

    if args.max_cases and args.max_cases < len(cases):
        cases = cases[:args.max_cases]
        print(f"Limited to {args.max_cases} cases")

    image_paths_list = [c["image_paths"] for c in cases]

    # Stage 1 & 2: Inference
    if not args.skip_inference:
        print("\n" + "=" * 60)
        print("Stage 1/3: nnUNet inference")
        print("=" * 60)
        nnunet_probs = run_nnunet_batch(
            image_paths_list, nnunet_out, args.nnunet_model_dir, device=args.device)

        print("\n" + "=" * 60)
        print("Stage 2/3: SwinUNETR inference")
        print("=" * 60)
        swin_probs = run_swin_batch(
            image_paths_list, swin_out, args.swin_checkpoint, args.plan_path, device=args.device)
    else:
        print("\nSkipping inference, loading cached predictions...")
        nnunet_probs = {}
        for c in cases:
            stem = Path(c["image_paths"][0]).name.replace(".nii.gz", "")
            npy_path = os.path.join(nnunet_out, f"{stem}_softmax_prob.npy")
            try:
                nnunet_probs[c["case_id"]] = np.load(npy_path).astype(np.float32)
            except FileNotFoundError:
                pass
        swin_probs = {}
        for c in cases:
            prob_path = os.path.join(swin_out, f"{c['case_id']}_prob.nii.gz")
            try:
                p = nib.load(prob_path).get_fdata().astype(np.float32)
                if p.ndim == 4 and p.shape[-1] == 4:
                    p = np.transpose(p, (3, 0, 1, 2))
                swin_probs[c["case_id"]] = p
            except (FileNotFoundError, nib.filebasedimages.ImageFileError):
                pass

    # Stage 3: Ensemble fusion + Dice computation
    print("\n" + "=" * 60)
    print("Stage 3/3: Ensemble fusion + Dice computation")
    print("=" * 60)

    dice_rows = []
    case_meta = {}  # lightweight: only file paths + dice values, no prob arrays

    for i, c in enumerate(cases):
        cid = c["case_id"]
        if cid not in nnunet_probs or cid not in swin_probs:
            print(f"[{i+1}/{len(cases)}] {cid}: SKIP (missing predictions)")
            continue

        ref_shape = nib.load(c["image_paths"][0]).shape

        ensemble_prob = compute_ensemble_probs(
            nnunet_probs[cid], swin_probs[cid], ref_shape,
            args.w_nnunet, args.w_swin)

        # Save ensemble prob to disk so it can be reloaded later without recomputation
        ens_prob_path = os.path.join(ens_out, f"{cid}_prob.npy")
        np.save(ens_prob_path, ensemble_prob)

        dice_dict = compute_all_dice(
            nnunet_probs[cid], swin_probs[cid], ensemble_prob,
            c["label_path"], ref_shape)

        row = {"case_id": cid}
        for m in MODEL_NAMES:
            for ci in range(3):
                row[f"{m}_d{ci+1}"] = dice_dict[m][ci]
        row["_ens_mean"] = _mean_dice(dice_dict["Ensemble"])
        dice_rows.append(row)

        # Store only lightweight metadata (file paths + dice), no prob arrays
        case_meta[cid] = {
            "dice_nnunet": dice_dict["nnUNet"],
            "dice_swin": dice_dict["SwinUNETR"],
            "dice_ens": dice_dict["Ensemble"],
            "image_paths": c["image_paths"],
            "label_path": c["label_path"],
        }

        # Free probability arrays from memory after processing
        del nnunet_probs[cid]
        del swin_probs[cid]

        if (i + 1) % 20 == 0 or (i + 1) == len(cases):
            print(f"[{i+1}/{len(cases)}] Processed dice scores")

    if not dice_rows:
        print("ERROR: No cases processed successfully")
        return

    # Save Dice CSV
    csv_path = os.path.join(args.output_dir, "dice_scores.csv")
    fieldnames = ["case_id"] + [f"{m}_d{ci+1}" for m in MODEL_NAMES for ci in range(3)]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in dice_rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"Saved: {csv_path}")

    # Save summary statistics CSV
    summary_path = os.path.join(args.output_dir, "dice_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "class", "mean", "std", "median", "min", "max", "valid_n"])
        for m in MODEL_NAMES:
            for ci, cls_name in enumerate(["necrotic", "edema", "enhancing"]):
                vals = [row[f"{m}_d{ci+1}"] for row in dice_rows
                        if not np.isnan(row[f"{m}_d{ci+1}"])]
                if vals:
                    writer.writerow([m, cls_name, np.mean(vals), np.std(vals),
                                     np.median(vals), np.min(vals), np.max(vals), len(vals)])
            means = []
            for row in dice_rows:
                class_vals = [row[f"{m}_d{ci+1}"] for ci in range(3)
                             if not np.isnan(row[f"{m}_d{ci+1}"])]
                if class_vals:
                    means.append(np.mean(class_vals))
            if means:
                writer.writerow([m, "mean", np.mean(means), np.std(means),
                                 np.median(means), np.min(means), np.max(means), len(means)])
    print(f"Saved: {summary_path}")

    # Select representative cases by ensemble mean Dice
    dice_rows.sort(key=lambda r: r["_ens_mean"])
    median_idx = len(dice_rows) // 2
    selected = {
        "median": dice_rows[median_idx],
        "best":   dice_rows[-1],
        "worst":  dice_rows[0],
    }

    def load_slice_data(row):
        """Reload volumes from disk for a single case (used for 3 viz cases only)."""
        cid = row["case_id"]
        meta = case_meta[cid]
        t1ce = nib.load(meta["image_paths"][1]).get_fdata().astype(np.float32)
        gt = nib.load(meta["label_path"]).get_fdata().astype(np.uint8)

        # Reload probs from disk
        nnunet_prob_path = os.path.join(
            nnunet_out,
            f"{Path(meta['image_paths'][0]).name.replace('.nii.gz', '')}_softmax_prob.npy")
        swin_prob_path = os.path.join(swin_out, f"{cid}_prob.nii.gz")
        ens_prob_path = os.path.join(ens_out, f"{cid}_prob.npy")

        nnunet_prob = np.load(nnunet_prob_path).astype(np.float32)
        swin_prob_raw = nib.load(swin_prob_path).get_fdata().astype(np.float32)
        if swin_prob_raw.ndim == 4 and swin_prob_raw.shape[-1] == 4:
            swin_prob_raw = np.transpose(swin_prob_raw, (3, 0, 1, 2))
        ensemble_prob = np.load(ens_prob_path).astype(np.float32)

        ref_shape = t1ce.shape
        seg_nnunet = _resample_seg_to_shape(
            np.argmax(nnunet_prob, axis=0).astype(np.uint8), ref_shape)
        seg_swin = _resample_seg_to_shape(
            np.argmax(swin_prob_raw, axis=0).astype(np.uint8), ref_shape)
        seg_ens = _resample_seg_to_shape(
            np.argmax(ensemble_prob, axis=0).astype(np.uint8), ref_shape)

        return {
            "case_id": cid, "t1ce_vol": t1ce, "gt_vol": gt,
            "seg_nnunet": seg_nnunet, "seg_swin": seg_swin, "seg_ens": seg_ens,
            "dice_nnunet": meta["dice_nnunet"],
            "dice_swin": meta["dice_swin"],
            "dice_ens": meta["dice_ens"],
        }

    print("\nGenerating figures...")

    median_slice = load_slice_data(selected["median"])
    generate_summary_figure(dice_rows, median_slice,
                            os.path.join(args.output_dir, "compare_summary.png"))

    best_slice = load_slice_data(selected["best"])
    generate_case_figure(best_slice, os.path.join(args.output_dir, "compare_best.png"),
                         tag="Best")

    worst_slice = load_slice_data(selected["worst"])
    generate_case_figure(worst_slice, os.path.join(args.output_dir, "compare_worst.png"),
                         tag="Worst")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary: Mean Dice (averaged over 3 tumor classes)")
    print("=" * 60)
    for m in MODEL_NAMES:
        all_means = []
        for row in dice_rows:
            vals = [row[f"{m}_d{ci+1}"] for ci in range(3)
                    if not np.isnan(row[f"{m}_d{ci+1}"])]
            if vals:
                all_means.append(np.mean(vals))
        if all_means:
            print(f"  {m:12s}: {np.mean(all_means):.4f} ± {np.std(all_means):.4f}  "
                  f"(median={np.median(all_means):.4f}, n={len(all_means)})")

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
