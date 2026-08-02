from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from ..io import read_json, require_numpy
from ..logger import get_logger
from .augmentation import augment_pair, build_augmentation
from .normalization import lesion_histogram

from monai.data import Dataset
from monai.transforms import Compose
import torch
import numpy as np

logger = get_logger(__name__)

def _load_patch(entry, *, prepared_dir: Path):
    """Load one prepared NPZ entry while retaining provenance fields."""
    np = require_numpy()
    result = dict(entry)
    with np.load(prepared_dir / entry["patch"]) as sample:
        result["image"] = sample["image"][0].astype(np.float32)
        result["mask"] = sample["mask"][0].astype(bool)
        result["histogram"] = sample["histogram"].astype(np.float32)
    return result


def _finalize_patch(sample, *, augmentation, seed: int, transform):
    """Optionally augment a patch and convert it to the model tensor contract."""
    image, mask, histogram = sample["image"], sample["mask"], sample["histogram"]
    if augmentation:
        rng = np.random.default_rng(
            seed + int(sample["_index"]) + int(torch.initial_seed() % 2**31)
        )
        image, mask = augment_pair(
            image, mask, augmentation, rng, transform=transform
        )
        histogram = lesion_histogram(image, mask, bins=len(histogram))
    return {
        "image": torch.from_numpy(np.asarray(image)[None].copy()).float(),
        "mask": torch.from_numpy(np.asarray(mask)[None].copy()),
        "histogram": torch.from_numpy(np.asarray(histogram).copy()).float(),
        "case_id": sample["case_id"],
        "component_id": sample["component_id"],
    }


def MeningitisPatchDataset(
    prepared_dir: str | Path,
    split: str,
    *,
    augmentation: dict[str, Any] | None = None,
    seed: int = 42,
):
    """Build a MONAI Dataset/Compose pipeline for one patient split."""
    prepared_dir = Path(prepared_dir)
    manifest = read_json(prepared_dir / "manifest.json")
    selected = [entry for entry in manifest["entries"] if entry["split"] == split]
    if not selected:
        raise ValueError(f"manifest has no entries for split={split}")
    entries = [{**entry, "_index": index} for index, entry in enumerate(selected)]
    augmentation = augmentation if augmentation and augmentation.get("enabled") else None
    transform = build_augmentation(augmentation) if augmentation else None
    pipeline = Compose(
        [
            partial(_load_patch, prepared_dir=prepared_dir),
            partial(
                _finalize_patch,
                augmentation=augmentation,
                seed=int(seed),
                transform=transform,
            ),
        ]
    )
    logger.info(
        "MeningitisPatchDataset: split=%s, entries=%d, augmentation=%s",
        split,
        len(entries),
        bool(augmentation),
    )
    return Dataset(entries, transform=pipeline)
