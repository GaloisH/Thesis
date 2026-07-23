"""Run a reproducible 3-D Poisson lesion-transplant experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    label,
)

try:
    from .poisson_blend import (
        copy_paste_3d,
        match_source_intensity,
        poisson_blend_3d,
        seam_metric,
    )
except ImportError:  # Allow direct ``python run_experiment.py`` execution.
    from poisson_blend import (  # type: ignore[no-redef]
        copy_paste_3d,
        match_source_intensity,
        poisson_blend_3d,
        seam_metric,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LESION_DIR = (
    PROJECT_ROOT / "src" / "synthesize" / "lesion_ldm" / "preprocess" / "0721"
)
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "datasets" / "nnUNet_raw" / "Dataset002_Meningitis"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "poisson_edit" / "multi_50"


def _case_id(path: Path) -> str:
    name = path.name
    if "_lesion" in name:
        return name.split("_lesion", 1)[0]
    if name.endswith(".nii.gz"):
        name = name[:-7]
    return name.rsplit("_", 1)[0]


def _read_array(path: Path) -> tuple[sitk.Image, np.ndarray]:
    image = sitk.ReadImage(str(path))
    return image, sitk.GetArrayFromImage(image)


def _write_like(array: np.ndarray, reference: sitk.Image, path: Path) -> None:
    output = sitk.GetImageFromArray(array)
    output.CopyInformation(reference)
    sitk.WriteImage(output, str(path), True)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    components, count = label(mask)
    if count == 0:
        raise ValueError("foreground extraction produced no connected component")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == int(np.argmax(sizes))


def build_foreground_mask(target_image: sitk.Image) -> np.ndarray:
    """Extract the high-intensity head/brain foreground using SimpleITK Otsu."""
    target = sitk.GetArrayFromImage(target_image).astype(np.float64)
    otsu_image = sitk.OtsuThreshold(target_image, 0, 1, 200)
    otsu = sitk.GetArrayFromImage(otsu_image).astype(bool)

    # SimpleITK's binary polarity depends on the threshold relation. Keep the
    # higher-intensity class, which is the useful foreground for these scans.
    mean_true = float(target[otsu].mean()) if otsu.any() else -np.inf
    mean_false = float(target[~otsu].mean()) if (~otsu).any() else -np.inf
    foreground = otsu if mean_true >= mean_false else ~otsu
    foreground = binary_closing(foreground, iterations=2)
    foreground = binary_fill_holes(foreground)
    return _largest_component(foreground)


def choose_placement(
    target: np.ndarray,
    target_image: sitk.Image,
    target_label: np.ndarray,
    lesion_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    max_attempts: int = 20_000,
    existing_mask: np.ndarray | None = None,
) -> tuple[int, int, int]:
    """Find a random in-foreground placement that avoids existing lesions."""
    foreground = build_foreground_mask(target_image)
    protected = binary_dilation(target_label > 0, iterations=5)
    if existing_mask is not None and existing_mask.any():
        protected = protected | binary_dilation(existing_mask > 0, iterations=5)
    lesion_halo = binary_dilation(lesion_mask, iterations=2)
    patch_shape = np.asarray(lesion_mask.shape, dtype=np.int64)
    target_shape = np.asarray(target.shape, dtype=np.int64)
    maximum_offset = target_shape - patch_shape
    if np.any(maximum_offset < 0):
        raise ValueError("lesion patch is larger than target volume")

    for _ in range(max_attempts):
        offset = tuple(
            int(rng.integers(0, maximum + 1)) for maximum in maximum_offset
        )
        roi = tuple(
            slice(start, start + size)
            for start, size in zip(offset, lesion_mask.shape)
        )
        if not np.all(foreground[roi][lesion_halo]):
            continue
        if np.any(protected[roi][lesion_mask]):
            continue
        target_ring = binary_dilation(lesion_mask, iterations=5) & ~binary_dilation(
            lesion_mask, iterations=1
        )
        if np.count_nonzero(target_ring) < 16:
            continue
        q25, q75 = np.percentile(target[roi][target_ring], (25, 75))
        if q75 - q25 <= 1e-6:
            continue
        return offset
    raise RuntimeError(f"no valid lesion placement found in {max_attempts} attempts")


def _inserted_mask(
    target_shape: tuple[int, int, int],
    lesion_mask: np.ndarray,
    offset: tuple[int, int, int],
    label_id: int = 1,
) -> np.ndarray:
    result = np.zeros(target_shape, dtype=np.uint8)
    roi = tuple(
        slice(start, start + size) for start, size in zip(offset, lesion_mask.shape)
    )
    result[roi][lesion_mask] = label_id
    return result


def save_comparison(
    original: np.ndarray,
    direct: np.ndarray,
    poisson: np.ndarray,
    inserted_mask: np.ndarray,
    output_path: Path,
) -> int:
    """Save full-slice and zoomed 2-D comparisons for the best axial slice."""
    mask_binary = inserted_mask > 0
    per_slice = mask_binary.sum(axis=(1, 2))
    z_index = int(np.argmax(per_slice))
    coordinates = np.argwhere(mask_binary[z_index] > 0)
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0) + 1
    padding = 20
    y0, x0 = max(0, y0 - padding), max(0, x0 - padding)
    y1 = min(original.shape[1], y1 + padding)
    x1 = min(original.shape[2], x1 + padding)

    values = original[z_index]
    vmin, vmax = np.percentile(values, (1.0, 99.5))
    volumes = (original, direct, poisson)
    titles = ("Original target", "Direct copy-paste", "3-D Poisson blend")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for column, (volume, title) in enumerate(zip(volumes, titles)):
        full_slice = np.rot90(volume[z_index])
        full_mask = np.rot90(mask_binary[z_index])
        axes[0, column].imshow(full_slice, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, column].contour(full_mask, levels=[0.5], colors="red", linewidths=0.8)
        axes[0, column].set_title(title)
        axes[0, column].axis("off")

        crop = volume[z_index, y0:y1, x0:x1]
        crop_mask = mask_binary[z_index, y0:y1, x0:x1]
        axes[1, column].imshow(np.rot90(crop), cmap="gray", vmin=vmin, vmax=vmax)
        axes[1, column].contour(
            np.rot90(crop_mask), levels=[0.5], colors="red", linewidths=1.2
        )
        axes[1, column].set_title(f"Zoomed ROI (z={z_index})")
        axes[1, column].axis("off")

    fig.suptitle(f"3-D lesion transplantation ({inserted_mask.max()} lesions): copy vs Poisson")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return z_index


def _select_sources(
    lesion_dir: Path,
    n_lesions: int,
    rng: np.random.Generator,
) -> list[tuple[Path, Path]]:
    """Select *n_lesions* valid lesion image/mask pairs."""
    candidates = list(lesion_dir.glob("*_mask.nii.gz"))
    rng.shuffle(candidates)

    selected: list[tuple[Path, Path]] = []
    for mask_path in candidates:
        if len(selected) >= n_lesions:
            break
        image_path = mask_path.with_name(
            mask_path.name.replace("_mask.nii.gz", "_img.nii.gz")
        )
        if not image_path.exists():
            continue
        _, mask = _read_array(mask_path)
        binary_mask = mask > 0
        if not binary_mask.any():
            continue
        if any(
            np.any(np.take(binary_mask, (0, -1), axis=axis)) for axis in range(3)
        ):
            continue
        selected.append((image_path, mask_path))

    if len(selected) < n_lesions:
        raise RuntimeError(
            f"only {len(selected)} valid lesion pairs found, need {n_lesions}"
        )
    return selected


def _select_target(
    images_dir: Path,
    labels_dir: Path,
    source_cases: set[str],
    channel: int,
    rng: np.random.Generator,
    target_image: Path | None,
) -> tuple[Path, Path]:
    if target_image is not None:
        candidates = [target_image.resolve()]
    else:
        candidates = list(images_dir.glob(f"*_{channel:04d}.nii.gz"))
        rng.shuffle(candidates)

    for image_path in candidates:
        case = _case_id(image_path)
        label_path = labels_dir / f"{case}.nii.gz"
        if case not in source_cases and label_path.exists():
            return image_path, label_path
    raise RuntimeError("no target case different from all source cases was found")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)

    source_pairs = _select_sources(args.lesion_dir, args.n_lesions, rng)
    source_cases = {_case_id(mask_path) for _, mask_path in source_pairs}
    target_image_path, target_label_path = _select_target(
        args.images_dir,
        args.labels_dir,
        source_cases,
        args.channel,
        rng,
        args.target_image,
    )

    target_image_sitk, target_raw = _read_array(target_image_path)
    _, target_label = _read_array(target_label_path)
    target = target_raw.astype(np.float64)

    if target.shape != target_label.shape:
        raise ValueError("target image and label shapes do not match")

    # Phase 1: find non-overlapping placements for all lesions.
    cumulative_placed = np.zeros(target.shape, dtype=np.uint8)
    placements: list[tuple[int, int, int]] = []
    lesion_records: list[dict[str, Any]] = []

    for idx, (src_img_path, src_mask_path) in enumerate(source_pairs):
        _, source = _read_array(src_img_path)
        _, source_mask_raw = _read_array(src_mask_path)
        source_mask = source_mask_raw > 0

        existing = cumulative_placed if cumulative_placed.any() else None
        offset = choose_placement(
            target,
            target_image_sitk,
            target_label,
            source_mask,
            rng,
            max_attempts=args.max_attempts,
            existing_mask=existing,
        )
        placements.append(offset)
        cumulative_placed = _inserted_mask(
            target.shape, source_mask, offset, label_id=idx + 1
        ) | (cumulative_placed > 0).astype(np.uint8)

        lesion_records.append(
            {
                "index": idx,
                "source_case": _case_id(src_mask_path),
                "source_image": str(src_img_path),
                "source_mask": str(src_mask_path),
                "offset_zyx": list(offset),
                "patch_shape_zyx": list(source.shape),
                "mask_voxels": int(source_mask.sum()),
            }
        )

    # Phase 2: apply all blends sequentially.
    direct = target.copy()
    poisson = target.copy()
    solver_stats_list: list[dict[str, Any]] = []
    intensity_stats_list: list[dict[str, Any]] = []

    for idx, ((src_img_path, src_mask_path), offset) in enumerate(
        zip(source_pairs, placements)
    ):
        _, source = _read_array(src_img_path)
        _, source_mask_raw = _read_array(src_mask_path)
        source_mask = source_mask_raw > 0
        roi = tuple(
            slice(start, start + size) for start, size in zip(offset, source.shape)
        )

        matched_source, int_stats = match_source_intensity(
            source, target[roi], source_mask
        )
        direct = copy_paste_3d(matched_source, direct, source_mask, offset)
        poisson, solver_stats = poisson_blend_3d(
            matched_source, poisson, source_mask, offset
        )
        solver_stats_list.append(solver_stats)
        intensity_stats_list.append(int_stats)

    # Build the final multi-label mask for visualization.
    full_mask = np.zeros(target.shape, dtype=np.uint8)
    for idx, ((_, src_mask_path), offset) in enumerate(
        zip(source_pairs, placements)
    ):
        _, source_mask_raw = _read_array(src_mask_path)
        source_mask_arr = source_mask_raw > 0
        full_mask = _inserted_mask(
            target.shape, source_mask_arr, offset, label_id=idx + 1
        ) | full_mask

    args.output_dir.mkdir(parents=True, exist_ok=True)
    poisson_path = args.output_dir / "poisson_blended.nii.gz"
    direct_path = args.output_dir / "copy_paste.nii.gz"
    mask_path = args.output_dir / "inserted_mask.nii.gz"
    comparison_path = args.output_dir / "comparison.png"
    metadata_path = args.output_dir / "metadata.json"

    _write_like(poisson.astype(np.float32), target_image_sitk, poisson_path)
    _write_like(direct.astype(np.float32), target_image_sitk, direct_path)
    _write_like(full_mask, target_image_sitk, mask_path)
    z_index = save_comparison(target, direct, poisson, full_mask, comparison_path)

    metadata: dict[str, Any] = {
        "seed": int(args.seed),
        "channel": int(args.channel),
        "n_lesions": args.n_lesions,
        "target_case": _case_id(target_image_path),
        "target_image": str(target_image_path),
        "target_label": str(target_label_path),
        "total_inserted_voxels": int((full_mask > 0).sum()),
        "visualized_axial_slice": z_index,
        "seam_metric": {
            "direct_copy_paste": seam_metric(direct, full_mask > 0, (0, 0, 0)),
            "poisson": seam_metric(poisson, full_mask > 0, (0, 0, 0)),
        },
        "lesions": lesion_records,
        "intensity_matching": intensity_stats_list,
        "poisson_solver": solver_stats_list,
        "outputs": {
            "poisson": str(poisson_path),
            "copy_paste": str(direct_path),
            "mask": str(mask_path),
            "comparison": str(comparison_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transplant a 3-D lesion patch with Poisson image editing."
    )
    parser.add_argument("--lesion-dir", type=Path, default=DEFAULT_LESION_DIR)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_DATASET_DIR / "imagesTr")
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_DATASET_DIR / "labelsTr")
    parser.add_argument("--channel", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-image", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-attempts", type=int, default=20_000)
    parser.add_argument("--n-lesions", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = run_experiment(args)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
