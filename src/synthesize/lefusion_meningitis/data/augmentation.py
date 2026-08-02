from __future__ import annotations

import math
from typing import Any

import numpy as np
from monai.transforms import Compose, Rand3DElasticd, RandAffined, RandFlipd


def build_augmentation(config: dict[str, Any]):
    """Build synchronized MONAI transforms applied jointly to image and mask."""
    # Both keys receive the same random transform so the mask stays aligned with the image.
    keys = ("image", "mask")
    rotation = math.radians(float(config["max_rotation_deg"]))
    scale_min, scale_max = (float(value) for value in config["scale_range"])
    if scale_min <= 0 or scale_max < scale_min:
        raise ValueError("augmentation.scale_range must contain increasing positive values")

    # Random flip along each axis independently (50% chance per axis).
    transforms = [RandFlipd(keys=keys, spatial_axis=axis, prob=0.5) for axis in range(3)]

    # Random rotation + scaling. scale_range is expressed as factors (e.g. 0.9..1.1),
    # so convert to MONAI's signed ranges. Image interpolates bilinearly; the mask uses
    # nearest so it stays binary.
    transforms.append(
        RandAffined(
            keys=keys,
            prob=1.0,
            rotate_range=(rotation,) * 3,
            scale_range=((1.0 - scale_min, scale_max - 1.0),) * 3,
            mode=("bilinear", "nearest"),
            padding_mode=("reflection", "zeros"),
        )
    )

    # Optional elastic deformation, disabled when elastic_alpha is 0.
    elastic_magnitude = float(config.get("elastic_alpha", 0.0))
    if elastic_magnitude > 0:
        elastic_sigma = float(config.get("elastic_sigma", 4.0))
        if elastic_sigma <= 0:
            raise ValueError("augmentation.elastic_sigma must be positive")
        transforms.append(
            Rand3DElasticd(
                keys=keys,
                prob=1.0,
                sigma_range=(elastic_sigma, elastic_sigma),
                magnitude_range=(elastic_magnitude, elastic_magnitude),
                mode=("bilinear", "nearest"),
                padding_mode=("reflection", "zeros"),
            )
        )
    return Compose(transforms)


def augment_pair(image, mask, config: dict[str, Any], rng, *, transform=None):
    """Apply MONAI augmentation and reject invalid transformed lesion masks."""
    original_image = np.asarray(image, dtype=np.float32)
    original_mask = np.asarray(mask, dtype=bool)
    if original_image.shape != original_mask.shape or original_image.ndim != 3:
        raise ValueError("augmentation expects aligned 3D image and mask arrays")
    pipeline = transform or build_augmentation(config)

    # Retry with a fresh RNG seed until the transformed mask stays valid.
    for _ in range(int(config.get("max_attempts", 8))):
        pipeline.set_random_state(seed=int(rng.integers(0, 2**32 - 1)))
        # Add a channel axis (MONAI expects (C, D, H, W)). The image gets a defensive
        # copy because np.asarray may alias the caller's array; the mask's astype
        # already returns a fresh array, so it needs no extra copy.
        result = pipeline(
            {
                "image": original_image[None].copy(),
                "mask": original_mask.astype(np.float32)[None],
            }
        )
        transformed_image = np.asarray(result["image"])[0].astype(np.float32)
        transformed_mask = np.asarray(result["mask"])[0] > 0.5
        # Reject masks that are too small to be a meaningful lesion...
        if transformed_mask.sum() < 8:
            continue
        # ...or that touch the patch border (lesion must stay fully inside).
        if any(
            np.any(np.take(transformed_mask, (0, -1), axis=axis))
            for axis in range(3)
        ):
            continue
        return transformed_image, transformed_mask
    raise RuntimeError("augmentation failed to produce a valid lesion mask")
