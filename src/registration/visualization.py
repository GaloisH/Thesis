"""Visualization utilities for registration outputs."""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import registration_core as reg


def pick_center(vol: np.ndarray) -> tuple[int, int, int]:
    nz = np.argwhere(vol > 0)
    if nz.size == 0:
        return tuple(np.array(vol.shape) // 2)
    return tuple(np.median(nz, axis=0).astype(int))


def save_plot(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_slices(vol: np.ndarray, title: str, path: str, cmap: str = "viridis",
                vmin=None, vmax=None) -> None:
    z, y, x = pick_center(vol)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slices = [(np.rot90(vol[z]), f"Axial z={z}"),
              (np.rot90(vol[:, y]), f"Coronal y={y}"),
              (np.rot90(vol[..., x]), f"Sagittal x={x}")]
    for ax, (img, lbl) in zip(axes, slices):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(lbl)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    save_plot(fig, path)


def plot_overlay(prob: np.ndarray, template: np.ndarray, path: str, alpha=0.55) -> None:
    z, y, x = pick_center(prob)
    vmax = np.percentile(prob, 99.5) if prob.any() else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slices = [(np.rot90(template[z]), np.rot90(prob[z]), f"Axial z={z}"),
              (np.rot90(template[:, y]), np.rot90(prob[:, y]), f"Coronal y={y}"),
              (np.rot90(template[..., x]), np.rot90(prob[..., x]), f"Sagittal x={x}")]
    for ax, (base, heat, lbl) in zip(axes, slices):
        ax.imshow(base, cmap="gray")
        ax.imshow(np.ma.masked_where(heat <= 0, heat), cmap="turbo", alpha=alpha,
                  vmin=0.0, vmax=max(vmax, 1e-6))
        ax.set_title(lbl)
        ax.axis("off")
    fig.suptitle("Lesion Probability Overlay on MNI")
    save_plot(fig, path)


def plot_histogram(prob: np.ndarray, path: str) -> None:
    nz = prob[prob > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    if nz.size == 0:
        ax.text(0.5, 0.5, "No non-zero voxels", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(nz, bins=50, color="#1f77b4", alpha=0.85)
        ax.set_xlabel("Probability")
        ax.set_ylabel("Voxel Count")
        ax.set_title("Probability Distribution")
        ax.grid(alpha=0.25)
    save_plot(fig, path)


def plot_regions_bar(csv_path: str, path: str, top_k=20) -> None:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"name": row.get("region_name", ""), "prob": float(row.get("mean_probability", 0))})
    if not rows:
        return
    rows = sorted(rows, key=lambda x: x["prob"], reverse=True)[:top_k]
    names = [r["name"] for r in rows][::-1]
    probs = [r["prob"] for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, int(top_k * 0.35))))
    ax.barh(names, probs, color="#2a9d8f", alpha=0.9)
    ax.set_xlabel("Mean Probability")
    ax.set_title(f"Top {len(rows)} Region Probabilities")
    ax.grid(axis="x", alpha=0.25)
    save_plot(fig, path)


def generate_region_heatmap_nii(csv_path: str, atlas_path: str, output_path: str,
                                 ref_img: nib.Nifti1Image | None = None) -> nib.Nifti1Image:
    """Generate 3D heatmap where each voxel's value is the region's mean probability.

    Args:
        csv_path: Path to region_probability_by_atlas.csv
        atlas_path: Path to atlas NIfTI file
        output_path: Path to save the heatmap NIfTI
        ref_img: Reference image (MNI template) to resample heatmap to match its space

    Returns:
        heatmap NIfTI image resampled to match reference image space
    """
    atlas_img = nib.load(atlas_path)
    atlas_data = np.rint(atlas_img.get_fdata()).astype(np.int32)

    # Load region probabilities from CSV
    region_probs = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = int(row.get("region_id", 0))
            prob = float(row.get("mean_probability", 0))
            region_probs[rid] = prob

    # Create heatmap by filling each region with its probability
    heatmap = np.zeros_like(atlas_data, dtype=np.float32)
    for rid, prob in region_probs.items():
        heatmap[atlas_data == rid] = prob

    # Create NIfTI with atlas's spatial information
    heatmap_img = nib.Nifti1Image(heatmap, atlas_img.affine, atlas_img.header)

    # Resample to match reference image (MNI template) if provided
    if ref_img is not None:
        from nibabel.processing import resample_from_to
        heatmap_img = resample_from_to(heatmap_img, (ref_img.shape, ref_img.affine), order=1)

    # Save as NIfTI
    nib.save(heatmap_img, output_path)
    return heatmap_img


def plot_region_heatmap_overlay(heatmap: np.ndarray, template: np.ndarray, path: str,
                                 alpha: float = 0.55) -> None:
    """Plot region probability heatmap overlay on MNI template (same style as lesion overlay)."""
    z, y, x = pick_center(heatmap)
    vmax = np.percentile(heatmap[heatmap > 0], 99) if heatmap.any() else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slices = [(np.rot90(template[z]), np.rot90(heatmap[z]), f"Axial z={z}"),
              (np.rot90(template[:, y]), np.rot90(heatmap[:, y]), f"Coronal y={y}"),
              (np.rot90(template[..., x]), np.rot90(heatmap[..., x]), f"Sagittal x={x}")]
    for ax, (base, heat, lbl) in zip(axes, slices):
        ax.imshow(base, cmap="gray")
        ax.imshow(np.ma.masked_where(heat <= 0, heat), cmap="turbo", alpha=alpha,
                  vmin=0.0, vmax=max(vmax, 1e-6))
        ax.set_title(lbl)
        ax.axis("off")
    fig.suptitle("Region Probability Heatmap on MNI")
    save_plot(fig, path)


def plot_regions_lollipop_top(csv_path: str, path: str, top_k: int = 30) -> None:
    """Vertical lollipop chart of top K high-risk brain regions."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "name": row.get("region_name", ""),
                "prob": float(row.get("mean_probability", 0)),
            })
    if not rows:
        return
    rows = sorted(rows, key=lambda x: x["prob"], reverse=True)[:top_k]
    probs = [r["prob"] for r in rows]
    names = [r["name"] for r in rows]

    # Color gradient from cool blue to hot red based on probability
    norm = plt.Normalize(min(probs), max(probs))
    colors = plt.cm.plasma(norm(probs))

    fig, ax = plt.subplots(figsize=(12, max(8, int(top_k * 0.32))))
    ax.hlines(y=range(len(names)), xmin=0, xmax=probs, colors=colors, linewidths=1.8, alpha=0.8)
    ax.scatter(probs, range(len(names)), c=colors, s=80, edgecolors="white", linewidths=0.5, zorder=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean Probability", fontsize=12)
    ax.set_title(f"Top {top_k} Brain Regions by Lesion Probability", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(probs) * 1.08)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_plot(fig, path)


def plot_regions_lollipop_distribution(csv_path: str, path: str) -> None:
    """Horizontal lollipop chart of all brain regions sorted by probability."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "id": int(row.get("region_id", 0)),
                "name": row.get("region_name", ""),
                "prob": float(row.get("mean_probability", 0)),
            })
    if not rows:
        return
    rows = sorted(rows, key=lambda x: x["prob"], reverse=True)
    n = len(rows)

    probs = [r["prob"] for r in rows]
    x_pos = range(n)
    norm = plt.Normalize(min(probs), max(probs))
    colors = plt.cm.plasma(norm(probs))

    fig, ax = plt.subplots(figsize=(14, max(10, int(n * 0.25))))
    ax.vlines(x=x_pos, ymin=0, ymax=probs, colors=colors, linewidths=1.2, alpha=0.7)
    ax.scatter(x_pos, probs, c=colors, s=30, edgecolors="none", zorder=3)
    ax.set_xticks(x_pos[::5])
    ax.set_xticklabels([rows[i]["name"] for i in range(0, n, 5)], fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Mean Probability", fontsize=12)
    ax.set_title(f"All {n} Brain Regions — Lesion Probability Distribution", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(probs) * 1.06)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_plot(fig, path)


def plot_qc_overlay(case_id: str, t1_path: str, mask_path: str, path: str) -> None:
    t1 = nib.load(t1_path).get_fdata()
    mask = nib.load(mask_path).get_fdata() > 0.5
    z, y, x = pick_center(mask.astype(np.float32))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slices = [(np.rot90(t1[z]), np.rot90(mask[z]), f"Axial z={z}"),
              (np.rot90(t1[:, y]), np.rot90(mask[:, y]), f"Coronal y={y}"),
              (np.rot90(t1[..., x]), np.rot90(mask[..., x]), f"Sagittal x={x}")]
    for ax, (img, seg, lbl) in zip(axes, slices):
        ax.imshow(img, cmap="gray")
        if seg.any():
            ax.contour(seg.astype(np.float32), levels=[0.5], colors="lime", linewidths=0.8)
        ax.set_title(lbl)
        ax.axis("off")
    fig.suptitle(f"QC: {case_id}")
    save_plot(fig, path)


def find_t1_path(dir: str, case_id: str) -> str | None:
    pref = os.path.join(dir, f"{case_id}_{reg.DEFAULT_CONFIG.modality.lower()}_mni.nii.gz")
    if os.path.exists(pref):
        return pref
    candidates = [p for p in glob.glob(os.path.join(dir, f"{case_id}_*_mni.nii.gz"))
                  if not p.endswith("_mask_mni.nii.gz")]
    return sorted(candidates)[0] if candidates else None


def generate_viz(output_dir: str, viz_dir: str, max_qc: int = 8,
                  atlas_path: str | None = None) -> None:
    os.makedirs(viz_dir, exist_ok=True)

    prob_path = os.path.join(output_dir, "lesion_probability_map.nii.gz")
    freq_path = os.path.join(output_dir, "lesion_frequency_map.nii.gz")

    if not os.path.exists(prob_path) or not os.path.exists(freq_path):
        raise FileNotFoundError("Maps not found. Run registration first.")

    prob = nib.load(prob_path).get_fdata().astype(np.float32)
    freq = nib.load(freq_path).get_fdata().astype(np.float32)
    template = nib.load(reg.resolve_mni_template()).get_fdata().astype(np.float32)

    plot_slices(freq, "Lesion Frequency Map", os.path.join(viz_dir, "01_frequency.png"), cmap="magma")
    plot_slices(prob, "Lesion Probability Map", os.path.join(viz_dir, "02_probability.png"), vmin=0, vmax=1)
    plot_overlay(prob, template, os.path.join(viz_dir, "03_overlay.png"))
    plot_histogram(prob, os.path.join(viz_dir, "04_histogram.png"))

    csv_path = os.path.join(output_dir, "region_probability_by_atlas.csv")
    if os.path.exists(csv_path):
        plot_regions_bar(csv_path, os.path.join(viz_dir, "05_regions.png"))
        plot_regions_lollipop_top(csv_path, os.path.join(viz_dir, "07_lollipop_top.png"))
        plot_regions_lollipop_distribution(csv_path, os.path.join(viz_dir, "08_lollipop_distribution.png"))
        # Generate region probability heatmap
        if atlas_path is None:
            atlas_path = os.path.join(output_dir, "atlas_ready", "AAL3v2.nii.gz")
        if os.path.exists(atlas_path):
            heatmap_path = os.path.join(output_dir, "region_probability_heatmap.nii.gz")
            # Load MNI template as Nifti1Image for resampling reference
            template_img = nib.load(reg.resolve_mni_template())
            heatmap_img = generate_region_heatmap_nii(csv_path, atlas_path, heatmap_path, ref_img=template_img)
            plot_region_heatmap_overlay(heatmap_img.get_fdata().astype(np.float32), template,
                                        os.path.join(viz_dir, "06_region_heatmap.png"))
        else:
            print(f"Atlas not found: {atlas_path}, skip region heatmap")

    reg_dir = os.path.join(output_dir, "registered")
    masks = sorted(glob.glob(os.path.join(reg_dir, "*_mask_mni.nii.gz")))
    if not masks:
        print("No registered masks for QC")
        return

    for mask_path in masks[:max_qc]:
        case_id = Path(mask_path).name.removesuffix("_mask_mni.nii.gz")
        t1_path = find_t1_path(reg_dir, case_id)
        if t1_path:
            plot_qc_overlay(case_id, t1_path, mask_path, os.path.join(viz_dir, f"qc_{case_id}.png"))
        else:
            print(f"Skip QC {case_id}: no T1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize registration outputs")
    parser.add_argument("--output-dir", default=reg.DEFAULT_CONFIG.output_dir)
    parser.add_argument("--viz-dir", default="")
    parser.add_argument("--max-qc", type=int, default=8)
    parser.add_argument("--atlas-path", default=None, help="Path to atlas NIfTI for region heatmap")
    args = parser.parse_args()

    viz_dir = args.viz_dir or os.path.join(args.output_dir, "visualizations")
    print("Generating visualizations...")
    generate_viz(args.output_dir, viz_dir, args.max_qc, args.atlas_path)
    print(f"Done: {viz_dir}")


if __name__ == "__main__":
    main()
