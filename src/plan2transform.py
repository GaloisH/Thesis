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
    Orientationd,
    Spacingd,
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
from monai.transforms import MapTransform
import torch


class ConvertToBratsRegionsd(MapTransform):
    """
    将单通道整数 label (1/2/3) 转换为 3 个二值化通道:
      - ch0 WT (Whole Tumor) : label ∈ {1, 2, 3}
      - ch1 TC (Tumor Core)  : label ∈ {2, 3}
      - ch2 ET (Enhancing)   : label == 3

    必须在所有空间变换和随机裁剪完成后调用，
    因为 RandCropByPosNegLabeld 需要单通道整数 label。
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]          # shape: (1, D, H, W), dtype: int/float
            wt = (label == 1) | (label == 2) | (label == 3)
            tc = (label == 2) | (label == 3)
            et = (label == 3)
            d[key] = torch.cat([wt, tc, et], dim=0).float()   # (3, D, H, W)
        return d


def _parse_plan(plan_path: str | Path) -> dict:
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    cfg = plan["configurations"]["3d_fullres"]

    data_order = cfg["resampling_fn_data_kwargs"].get("order", 3)
    interp_image = "bilinear" if data_order >= 2 else "nearest"
    interp_label = "nearest"

    # use_mask_for_norm: nnUNet 的 ZScoreNormalization 仅在前景体素上做统计
    use_mask = any(cfg.get("use_mask_for_norm", [False]))

    return {
        "spacing": cfg["spacing"],
        "patch_size": cfg["patch_size"],
        "batch_size": cfg["batch_size"],
        "use_mask_for_norm": use_mask,
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

    # ------------------------------------------------------------------
    # base_transforms:
    #   1. 加载 + channel 处理
    #   2. 空间对齐 (orientation / spacing)
    #   3. 前景裁剪
    #   4. 归一化 —— 仅 Z-score，与 nnUNet ZScoreNormalization 一致
    #      不做 percentile 0-1 缩放（那会与 Z-score 叠加，偏离 nnUNet 方案）
    # ------------------------------------------------------------------
    base_transforms = [
        # image: 4 个路径 → (4, D, H, W)；label: 1 个路径 → (1, D, H, W)
        LoadImaged(keys=keys, image_only=True, ensure_channel_first=True),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=p["spacing"],
            mode=(p["interp_image"], p["interp_label"]),
        ),
        CropForegroundd(keys=keys, source_key=image_key, allow_smaller=True),
        # nnUNet ZScoreNormalization: (x - mean) / std，仅在非零前景体素上统计
        NormalizeIntensityd(
            keys=[image_key],
            nonzero=p["use_mask_for_norm"],   # True → 只用前景体素的 mean/std
            channel_wise=True,                 # 4 个模态分别归一化
        ),
    ]

    # ------------------------------------------------------------------
    # train_transforms:
    #   空间 pad → 随机裁剪（label 仍是单通道整数，采样逻辑正确）
    #   → ConvertToBratsRegionsd（裁剪完成后再转多通道）
    #   → 数据增强 → ToTensor
    # ------------------------------------------------------------------
    train_transforms = Compose(
        base_transforms + [
            SpatialPadd(keys=keys, spatial_size=p["patch_size"], mode="constant"),
            RandCropByPosNegLabeld(
                keys=keys,
                label_key=label_key,
                spatial_size=p["patch_size"],
                pos=1,
                neg=1,
                num_samples=p["batch_size"],   # 每张图随机采 batch_size 个 patch
                image_key=image_key,
                image_threshold=0,
            ),
            # 空间变换完成后，将整数 label 转为 BraTS 三区域多通道二值图
            ConvertToBratsRegionsd(keys=[label_key]),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=keys, prob=0.5, max_k=3),
            RandScaleIntensityd(keys=[image_key], factors=0.1, prob=0.1),
            RandShiftIntensityd(keys=[image_key], offsets=0.1, prob=0.1),
            ToTensord(keys=keys),
        ]
    )

    # ------------------------------------------------------------------
    # val_transforms:
    #   不做随机裁剪；转换区域标签后直接 ToTensor
    # ------------------------------------------------------------------
    val_transforms = Compose(
        base_transforms + [
            ConvertToBratsRegionsd(keys=[label_key]),
            ToTensord(keys=keys),
        ]
    )

    return train_transforms, val_transforms


if __name__ == "__main__":
    plan_path = r"D:\python_code\projects\thesis\datasets\nnUNet_preprocessed\Dataset101_Meningioma\nnUNetPlans.json"
    train_t, val_t = build_transforms_from_plan(plan_path)
    print("Train Transforms:")
    print(train_t)
    print("\nValidation Transforms:")
    print(val_t)
