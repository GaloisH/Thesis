"""
nnunet_to_monai_transforms.py
Reads nnUNetPlans.json to automatically extract 3d_fullres configurations
and returns (train_transforms, val_transforms) MONAI Compose objects.
"""
import json
from pathlib import Path
from typing import Tuple

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
    ToTensord,
)

def _parse_plan(plan_path: str | Path) -> dict:
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    cfg = plan["configurations"]["3d_fullres"]
    
    # Defaults and percentiles used by nnUNet
    LOWER_PCT, UPPER_PCT = 0.5, 99.5
    
    # Resampling modes
    data_order = cfg["resampling_fn_data_kwargs"].get("order", 3)
    seg_order = cfg["resampling_fn_seg_kwargs"].get("order", 0)
    
    interp_image = "bilinear" if data_order >= 2 else "nearest"
    interp_label = "nearest"  # label is always nearest

    return {
        "spacing": cfg["spacing"],
        "patch_size": cfg["patch_size"],
        "batch_size": cfg["batch_size"],
        "use_mask_for_norm": any(cfg.get("use_mask_for_norm", [False])),
        "percentile_low": LOWER_PCT,
        "percentile_high": UPPER_PCT,
        "interp_image": interp_image,
        "interp_label": interp_label,
    }

def build_transforms_from_plan(
    plan_path: str | Path,
    image_key: str = "image",
    label_key: str = "label",
) -> Tuple[Compose, Compose]:
    p = _parse_plan(plan_path)
    keys = [image_key, label_key]

    base_transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=p["spacing"],
            mode=(p["interp_image"], p["interp_label"]),
        ),
        CropForegroundd(keys=keys, source_key=image_key, allow_smaller=True),
        ScaleIntensityRangePercentilesd(
            keys=[image_key],
            lower=p["percentile_low"],
            upper=p["percentile_high"],
            b_min=0.0,
            b_max=1.0,
            clip=True,
            relative=False,
        ),
        NormalizeIntensityd(
            keys=[image_key],
            nonzero=p["use_mask_for_norm"],
            channel_wise=True,
        ),
    ]

    train_transforms = Compose(
        base_transforms + [
            SpatialPadd(keys=keys, spatial_size=p["patch_size"], mode="constant"),
            RandCropByPosNegLabeld(
                keys=keys,
                label_key=label_key,
                spatial_size=p["patch_size"],
                pos=1,
                neg=1,
                num_samples=p["batch_size"],
                image_key=image_key,
                image_threshold=0,
            ),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=keys, prob=0.5, max_k=3),
            RandScaleIntensityd(keys=[image_key], factors=0.1, prob=0.1),
            RandShiftIntensityd(keys=[image_key], offsets=0.1, prob=0.1),
            ToTensord(keys=keys),
        ]
    )

    val_transforms = Compose(base_transforms + [ToTensord(keys=keys)])

    return train_transforms, val_transforms

if __name__ == "__main__":
    plan_path = r"D:\python_code\projects\thesis\datasets\nnUNet_preprocessed\Dataset101_Meningioma\nnUNetPlans.json"
    train_t, val_t = build_transforms_from_plan(plan_path)
    print("Train Transforms:")
    print(train_t)
    print("\nValidation Transforms:")
    print(val_t)