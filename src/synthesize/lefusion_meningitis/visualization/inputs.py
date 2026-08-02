from __future__ import annotations

from pathlib import Path

from ..io import image_path, label_path, read_json, require_nibabel, require_numpy


def case_id_from_path(path: str | Path) -> str:
    """Infer a case ID from a NIfTI path, stripping an optional channel suffix."""
    name = Path(path).name
    name = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
    if len(name) > 5 and name[-5] == "_" and name[-4:].isdigit():
        name = name[:-5]
    return name


def load_aligned_ras(image_path: str | Path, mask_path: str | Path):
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


def batch_entries(config):
    """Resolve the largest prepared lesion component for each train/validation case."""
    prepared_dir = Path(config["data"]["prepared_dir"])
    split_path = prepared_dir / "split.json"
    manifest_path = prepared_dir / "manifest.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"prepared split not found: {split_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prepared manifest not found: {manifest_path}")
    split = read_json(split_path)
    manifest = read_json(manifest_path)
    cases = list(
        dict.fromkeys(
            list(split.get("cases", {}).get("train", []))
            + list(split.get("cases", {}).get("val", []))
        )
    )
    if not cases:
        raise ValueError("prepared split contains no train or validation cases")
    candidates_by_case = {}
    for candidate in manifest.get("entries", []):
        candidate_case = candidate["case_id"]
        current = candidates_by_case.get(candidate_case)
        rank = (int(candidate["voxels"]), -int(candidate["component_id"]))
        if current is None or rank > (
            int(current["voxels"]),
            -int(current["component_id"]),
        ):
            candidates_by_case[candidate_case] = candidate
    missing = [case for case in cases if case not in candidates_by_case]
    if missing:
        raise ValueError(
            "train/validation cases have no valid prepared lesion component: "
            + ", ".join(missing)
        )
    dataset = Path(config["data"]["source_dataset"])
    channel = int(config["data"]["channel"])
    return [
        {
            "case_id": case,
            "image": image_path(dataset, case, channel),
            "mask": label_path(dataset, case),
            "prepared_entry": candidates_by_case[case],
        }
        for case in cases
    ]
