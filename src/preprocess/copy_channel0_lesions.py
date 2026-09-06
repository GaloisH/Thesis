"""Copy channel 0 lesion voxels into channels 1 and 2 of an nnUNet dataset.

Example:
    python src/preprocess/copy_channel0_lesions.py \
        datasets/nnUNet_raw/Dataset002_Meningitis
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


SOURCE_SUFFIX = "_0000.nii.gz"
TARGET_CHANNELS = (1, 2)


def copy_channel0_lesions(dataset_dir: Path) -> int:
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    source_paths = sorted(images_dir.glob(f"*{SOURCE_SUFFIX}"))
    if not source_paths:
        raise FileNotFoundError(f"No channel 0 images found in {images_dir}")

    cases = []
    for source_path in source_paths:
        case_id = source_path.name.removesuffix(SOURCE_SUFFIX)
        label_path = labels_dir / f"{case_id}.nii.gz"
        target_paths = [
            images_dir / f"{case_id}_{channel:04d}.nii.gz"
            for channel in TARGET_CHANNELS
        ]
        for path in [label_path, *target_paths]:
            if not path.is_file():
                raise FileNotFoundError(f"Missing file: {path}")

        reference = nib.load(source_path)
        for path in [label_path, *target_paths]:
            image = nib.load(path)
            if image.shape != reference.shape or not np.allclose(
                image.affine, reference.affine
            ):
                raise ValueError(f"Image geometry does not match channel 0: {path}")
        cases.append((source_path, label_path, target_paths))

    for source_path, label_path, target_paths in cases:
        source = np.asanyarray(nib.load(source_path).dataobj)
        lesion = np.asanyarray(nib.load(label_path).dataobj) > 0
        for target_path in target_paths:
            target_image = nib.load(target_path)
            output = np.asanyarray(target_image.dataobj).copy()
            output[lesion] = source[lesion]
            nib.save(
                nib.Nifti1Image(output, target_image.affine, target_image.header),
                target_path,
            )

    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy channel 0 lesion voxels into channels 1 and 2 in place."
    )
    parser.add_argument("dataset_dir", type=Path, help="nnUNet dataset root directory")
    args = parser.parse_args()
    count = copy_channel0_lesions(args.dataset_dir)
    print(f"Processed {count} case(s).")


if __name__ == "__main__":
    main()