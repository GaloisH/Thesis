"""Generate a FastSurfer ``subject_id=path_to_t1`` input list.

The source dataset, image channel, project root, and FastSurfer subjects
directory are all read from ``config/lefusion_meningitis.yaml`` (or a config
supplied with ``--config``). The list is written as ``subject_list.txt`` inside
the configured ``data.fastsurfer_subjects_dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "lefusion_meningitis.yaml"


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def load_paths(config_path: Path) -> tuple[Path, Path, int]:
    """Return the configured imagesTr directory, output list, and channel."""
    with config_path.resolve().open("r", encoding="utf-8") as stream:
        config: dict[str, Any] = yaml.safe_load(stream) or {}

    declared_root = Path(config.get("project_root", ".")).expanduser()
    project_root = (
        declared_root if declared_root.is_absolute() else PROJECT_ROOT / declared_root
    ).resolve()

    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("Missing mapping 'data' in config")

    required = ("source_dataset", "fastsurfer_subjects_dir")
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise ValueError(f"Missing data config value(s): {', '.join(missing)}")

    try:
        channel = int(data.get("channel", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("data.channel must be an integer") from exc
    if channel < 0:
        raise ValueError("data.channel must be non-negative")

    source_dataset = _resolve_path(str(data["source_dataset"]), project_root)
    subjects_dir = _resolve_path(str(data["fastsurfer_subjects_dir"]), project_root)
    output_path = subjects_dir / "subject_list.txt"
    return source_dataset / "imagesTr", output_path, channel


def collect_subjects(images_dir: Path, channel: int) -> list[tuple[str, Path]]:
    """Collect and sort one configured nnUNet image channel per subject."""
    if not images_dir.is_dir():
        raise FileNotFoundError(f"nnUNet imagesTr directory not found: {images_dir}")

    suffix = f"_{channel:04d}.nii.gz"
    images = sorted(images_dir.glob(f"*{suffix}"))
    if not images:
        raise FileNotFoundError(
            f"No channel {channel} NIfTI images matching '*{suffix}' in {images_dir}"
        )

    subjects: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for image_path in images:
        case_id = image_path.name[: -len(suffix)]
        if not case_id or "=" in case_id or "\n" in case_id or "\r" in case_id:
            raise ValueError(f"Invalid FastSurfer subject ID derived from {image_path.name!r}")
        if case_id in seen:
            raise ValueError(f"Duplicate subject ID: {case_id}")
        seen.add(case_id)
        subjects.append((case_id, image_path.resolve()))
    return subjects


def write_subject_list(subjects: Sequence[tuple[str, Path]], output_path: Path) -> None:
    """Write subjects in the format accepted by brun_fastsurfer.sh."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{subject_id}={image_path}" for subject_id, image_path in subjects]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a FastSurfer subject list from nnUNet imagesTr."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config path (default: {DEFAULT_CONFIG})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_dir, output_path, channel = load_paths(args.config)
    subjects = collect_subjects(images_dir, channel)
    write_subject_list(subjects, output_path)
    print(f"Wrote {len(subjects)} subjects to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
