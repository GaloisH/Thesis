"""Generate a FastSurfer ``subject_id=path_to_t1`` input list.

The input directory is expected to be flat, like ``datasets/IXI-T1``, with
one ``.nii.gz`` T1 image per subject. The generated list is saved as
``subject_list.txt`` in the requested output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


NIFTI_SUFFIX = ".nii.gz"
OUTPUT_FILENAME = "subject_list.txt"


def collect_subjects(input_dir: Path) -> list[tuple[str, Path]]:
    """Collect T1 images and derive subject IDs from their file names."""
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.name.endswith(NIFTI_SUFFIX)
    )
    if not images:
        raise FileNotFoundError(f"No '*{NIFTI_SUFFIX}' files found in: {input_dir}")

    subjects: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for image_path in images:
        subject_id = image_path.name[: -len(NIFTI_SUFFIX)]
        if not subject_id or any(char in subject_id for char in "=\n\r"):
            raise ValueError(f"Invalid subject ID derived from: {image_path.name!r}")
        if subject_id in seen:
            raise ValueError(f"Duplicate subject ID: {subject_id}")
        seen.add(subject_id)
        subjects.append((subject_id, image_path.resolve()))

    return subjects


def write_subject_list(subjects: Sequence[tuple[str, Path]], output_dir: Path) -> Path:
    """Write ``subject_list.txt`` and return its path."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    lines = [f"{subject_id}={image_path}" for subject_id, image_path in subjects]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a FastSurfer subject list from a flat folder of T1 NIfTI images."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing T1 .nii.gz files")
    parser.add_argument("output_dir", type=Path, help="Folder in which to save subject_list.txt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subjects = collect_subjects(args.input_dir)
    output_path = write_subject_list(subjects, args.output_dir)
    print(f"Wrote {len(subjects)} subjects to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
