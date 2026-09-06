"""Merge LeFusion synthetic outputs into an nnUNet v2 raw dataset.

The script intentionally has no command-line arguments.  Edit the constants in
the configuration section below when a different synthesis run or Dataset ID is
needed, then run::

    python src/preprocess/merge_syn_data.py

Only the cumulative labels in ``synthetic/labels`` are training targets.  The
files in ``synthetic/masks`` describe inserted voxels only and therefore must
not be used as nnUNet labels.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Hard-coded project configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = PROJECT_ROOT / "outputs" / "lefusion_meningitis" / "synthetic"
OUTPUT_DATASET_DIR = (
    PROJECT_ROOT / "datasets" / "nnUNet_raw" / "Dataset202_MeningitisLeFusion"
)
SOURCE_SPLITS_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "nnUNet_preprocessed"
    / "Dataset002_Meningitis"
    / "splits_final.json"
)
OUTPUT_SPLITS_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "nnUNet_preprocessed"
    / "Dataset202_MeningitisLeFusion"
    / "splits_final.json"
)
DATASET_NAME = "MeningitisLeFusion"
CHANNEL_NAMES = {"0": "T1"}
LABELS = {"background": 0, "meningitis": 1}
FILE_ENDING = ".nii.gz"


def _case_id_from_image(path: Path) -> str:
    suffix = "_0000.nii.gz"
    if not path.name.endswith(suffix):
        raise ValueError(
            f"invalid synthetic image name {path.name!r}; expected '*{suffix}'"
        )
    case_id = path.name[: -len(suffix)]
    if not case_id:
        raise ValueError(f"empty case ID in image name: {path}")
    return case_id


def _output_case_id(source_case_id: str) -> str:
    """Remove the synthesis marker from an exported nnUNet case ID."""
    case_id = source_case_id.removesuffix("_syn")
    if not case_id:
        raise ValueError(f"empty output case ID after removing '_syn': {source_case_id}")
    return case_id


def _discover_pairs(synthetic_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return validated ``(case_id, image, label)`` tuples."""
    images_dir = synthetic_dir / "images"
    labels_dir = synthetic_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"synthetic images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"synthetic labels directory not found: {labels_dir}")

    image_paths = sorted(images_dir.glob("*.nii.gz"), key=lambda path: path.name)
    if not image_paths:
        raise RuntimeError(f"no synthetic NIfTI images found in: {images_dir}")

    pairs: list[tuple[str, Path, Path]] = []
    seen: set[str] = set()
    for image_path in image_paths:
        case_id = _case_id_from_image(image_path)
        if case_id in seen:
            raise ValueError(f"duplicate synthetic case ID: {case_id}")
        label_path = labels_dir / f"{case_id}.nii.gz"
        if not label_path.is_file():
            raise FileNotFoundError(
                f"label missing for synthetic image {image_path.name}: {label_path}"
            )
        seen.add(case_id)
        pairs.append((case_id, image_path, label_path))

    orphan_labels = sorted(
        path.name
        for path in labels_dir.glob("*.nii.gz")
        if path.name[:-7] not in seen
    )
    if orphan_labels:
        raise ValueError(
            "synthetic labels without matching images: " + ", ".join(orphan_labels)
        )
    return pairs


def _read_fold_zero(source_file: Path, case_ids: list[str]) -> dict[str, list[str]]:
    """Filter Dataset002 fold 0 to the cases present in this synthetic dataset."""
    if not source_file.is_file():
        raise FileNotFoundError(f"source nnUNet split file not found: {source_file}")
    with source_file.open("r", encoding="utf-8") as file:
        splits = json.load(file)
    if not isinstance(splits, list) or not splits:
        raise ValueError(f"source split file contains no fold 0: {source_file}")

    fold_zero = splits[0]
    if not isinstance(fold_zero, dict) or not all(
        isinstance(fold_zero.get(key), list) for key in ("train", "val")
    ):
        raise ValueError(f"invalid fold 0 format in: {source_file}")

    requested = set(case_ids)
    train = [case_id for case_id in fold_zero["train"] if case_id in requested]
    val = [case_id for case_id in fold_zero["val"] if case_id in requested]
    overlap = set(train) & set(val)
    if overlap:
        raise ValueError(f"fold 0 train/val overlap: {sorted(overlap)}")
    missing = requested - set(train) - set(val)
    if missing:
        raise ValueError(
            "synthetic cases absent from Dataset002 fold 0: " + ", ".join(sorted(missing))
        )
    if not train or not val:
        raise ValueError(
            f"filtered fold 0 must contain both train and val cases; "
            f"got train={len(train)}, val={len(val)}"
        )
    return {"train": train, "val": val}


def _write_splits_file(output_file: Path, fold_zero: dict[str, list[str]]) -> None:
    """Write the single custom fold consumed by nnUNet when training with fold 0."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump([fold_zero], file, ensure_ascii=False, indent=4)
        file.write("\n")


def merge_synthetic_dataset(
    synthetic_dir: Path = SYNTHETIC_DIR,
    output_dataset_dir: Path = OUTPUT_DATASET_DIR,
    source_splits_file: Path = SOURCE_SPLITS_FILE,
    output_splits_file: Path = OUTPUT_SPLITS_FILE,
) -> dict[str, object]:
    """Create a new, synthetic-only nnUNet raw dataset.

    The destination must not already exist.  Files are first assembled in a
    sibling staging directory and published only after every copy succeeds.
    """
    synthetic_dir = Path(synthetic_dir).resolve()
    output_dataset_dir = Path(output_dataset_dir).resolve()
    pairs = _discover_pairs(synthetic_dir)

    output_case_ids = [_output_case_id(case_id) for case_id, _, _ in pairs]
    if len(output_case_ids) != len(set(output_case_ids)):
        raise ValueError("duplicate case IDs after removing the '_syn' suffix")
    fold_zero = _read_fold_zero(Path(source_splits_file).resolve(), output_case_ids)

    if output_dataset_dir.exists():
        raise FileExistsError(
            f"output dataset already exists: {output_dataset_dir}; "
            "remove it explicitly or choose another Dataset ID in the script"
        )
    output_dataset_dir.parent.mkdir(parents=True, exist_ok=True)

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dataset_dir.name}.building-",
            dir=output_dataset_dir.parent,
        )
    )
    try:
        images_tr = staging_dir / "imagesTr"
        labels_tr = staging_dir / "labelsTr"
        metadata_out = staging_dir / "metadata"
        images_tr.mkdir()
        labels_tr.mkdir()

        source_metadata = synthetic_dir / "metadata"
        if source_metadata.is_dir():
            metadata_out.mkdir()

        for (source_case_id, image_path, label_path), output_case_id in zip(
            pairs, output_case_ids
        ):
            shutil.copy2(image_path, images_tr / f"{output_case_id}_0000.nii.gz")
            shutil.copy2(label_path, labels_tr / f"{output_case_id}.nii.gz")
            metadata_path = source_metadata / f"{source_case_id}.json"
            if metadata_path.is_file():
                shutil.copy2(metadata_path, metadata_out / f"{output_case_id}.json")

        dataset_json = {
            "name": DATASET_NAME,
            "channel_names": CHANNEL_NAMES,
            "labels": LABELS,
            "numTraining": len(output_case_ids),
            "file_ending": FILE_ENDING,
        }
        with (staging_dir / "dataset.json").open("w", encoding="utf-8") as file:
            json.dump(dataset_json, file, ensure_ascii=False, indent=4)
            file.write("\n")

        output_dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.replace(output_dataset_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    _write_splits_file(Path(output_splits_file).resolve(), fold_zero)

    return {
        "input": str(synthetic_dir),
        "output": str(output_dataset_dir),
        "numTraining": len(pairs),
        "cases": output_case_ids,
        "fold_0_train": len(fold_zero["train"]),
        "fold_0_val": len(fold_zero["val"]),
        "splits_file": str(Path(output_splits_file).resolve()),
    }


def main() -> None:
    result = merge_synthetic_dataset()
    print(
        f"Created {result['output']} with "
        f"{result['numTraining']} synthetic training cases."
    )


if __name__ == "__main__":
    main()
