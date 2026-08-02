from __future__ import annotations

import numpy as np
import torch


def sample_histogram(histograms, rng, jitter: float):
    """对训练直方图库进行随机采样并可选地添加高斯扰动。"""
    if len(histograms) == 0:
        raise ValueError("histogram library is empty")
    histogram = np.asarray(
        histograms[int(rng.integers(0, len(histograms)))], dtype=np.float32
    ).copy()
    if jitter > 0:
        histogram += rng.normal(0.0, jitter, size=histogram.shape).astype(np.float32)
    histogram = np.clip(histogram, 0.0, None)
    total = float(histogram.sum())
    if total <= 0:
        raise ValueError("histogram perturbation produced an empty distribution")
    return histogram / total


def sample_composite_patch(
    model,
    background,
    mask,
    histogram,
    *,
    device,
    seed: int,
    brightness_margin: float = 0.2,
    brightness_transition_voxels: float = 3.0,
):
    """使用训练好的模型生成一个合成补丁，并将其与背景图像进行硬合成。"""
    background = np.asarray(background, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    histogram = np.asarray(histogram, dtype=np.float32)
    if background.shape != mask.shape:
        raise ValueError("background and mask patch shapes differ")
    if not mask.any():
        raise ValueError("synthesis mask is empty")

    background_tensor = torch.from_numpy(background[None, None]).to(device)
    mask_tensor = torch.from_numpy(mask[None, None]).to(device)
    histogram_tensor = torch.from_numpy(histogram[None]).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    generated_tensor = model.sample_patch(
        background_tensor,
        mask_tensor,
        histogram_tensor,
        generator=generator,
    )
    generated = generated_tensor[0, 0].detach().cpu().numpy()
    generated = brighten_lesion_interior(
        background,
        generated,
        mask,
        margin=brightness_margin,
        transition_voxels=brightness_transition_voxels,
    )
    return generated, hard_composite(background, generated, mask)


def brighten_lesion_interior(
    background,
    generated,
    mask,
    *,
    margin: float = 0.1,
    transition_voxels: float = 3.0,
):
    """Raise lesion intensity progressively from its boundary toward its center."""
    try:
        from scipy.ndimage import binary_dilation, distance_transform_edt
    except ImportError as exc:
        raise RuntimeError("SciPy is required for lesion brightening") from exc
    background = np.asarray(background, dtype=np.float32)
    adjusted = np.asarray(generated, dtype=np.float32).copy()
    mask = np.asarray(mask, dtype=bool)
    if background.shape != adjusted.shape or mask.shape != background.shape:
        raise ValueError("brightening inputs have incompatible shapes")
    if not mask.any():
        raise ValueError("brightening mask is empty")
    if margin < 0:
        raise ValueError("brightness margin must be non-negative")
    if transition_voxels <= 0:
        raise ValueError("brightness transition must be positive")
    ring_width = max(1, int(np.ceil(transition_voxels)))
    ring = binary_dilation(mask, iterations=ring_width) & ~mask
    if not ring.any():
        return adjusted
    background_level = float(np.percentile(background[ring], 90))
    lesion_level = float(np.percentile(adjusted[mask], 25))
    offset = max(0.0, background_level + float(margin) - lesion_level)
    if offset == 0:
        return adjusted
    distance = distance_transform_edt(mask)
    weight = np.clip(distance / float(transition_voxels), 0.0, 1.0)
    adjusted[mask] = np.clip(adjusted[mask] + offset * weight[mask], -1.0, 1.0)
    return adjusted


def hard_composite(background, generated, mask):
    """Replace only masked voxels and preserve every exterior voxel exactly."""
    background = np.asarray(background)
    generated = np.asarray(generated)
    mask = np.asarray(mask, dtype=bool)
    if background.shape != generated.shape or mask.shape != background.shape:
        raise ValueError("composite inputs have incompatible shapes")
    result = background.copy()
    result[mask] = generated[mask]
    return result
