from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def require_torch():
    """延迟导入 PyTorch，并在环境不完整时报告明确错误。"""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and the LeFusion requirements are required for model commands"
        ) from exc
    return torch


def load_official_unet():
    """Load the vendored architecture without modifying the official source tree."""
    vendor_root = Path(__file__).resolve().parents[1] / "LeFusion" / "LeFusion"
    if not vendor_root.exists():
        raise RuntimeError(f"vendored LeFusion source not found: {vendor_root}")
    value = str(vendor_root)
    if value not in sys.path:
        sys.path.insert(0, value)
    try:
        from ddpm.diffusion import Unet3D
    except ImportError as exc:
        raise RuntimeError(
            "unable to import the official LeFusion U-Net; install "
            "src/synthesize/LeFusion/requirements.txt"
        ) from exc
    return Unet3D


def _extract(values, timesteps, shape):
    """按 batch 时间步提取扩散系数并扩展到输入维度。"""
    torch = require_torch()
    selected = values.gather(0, timesteps)
    return selected.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1))).to(
        device=timesteps.device
    )


class LeFusionH:
    """Factory-backed nn.Module to keep import-time dependency errors actionable."""

    def __new__(cls, config: dict[str, Any]):
        """根据配置动态创建依赖 PyTorch 的 LeFusion-H 模型。"""
        torch = require_torch()
        nn = torch.nn
        Unet3D = load_official_unet()

        class _LeFusionH(nn.Module):
            def __init__(self):
                """初始化官方 3D U-Net 与 DDPM 前向/后验系数。"""
                super().__init__()
                self.config = dict(config)
                self.timesteps = int(config["timesteps"])
                self.loss_type = str(config.get("loss_type", "l1"))
                self.denoiser = Unet3D(
                    dim=int(config["image_size"]),
                    dim_mults=tuple(int(value) for value in config["dim_mults"]),
                    channels=int(config["channels"]),
                    cond_dim=int(config["histogram_dim"]),
                )
                betas = torch.linspace(1e-4, 2e-2, self.timesteps, dtype=torch.float64)
                alphas = 1.0 - betas
                cumulative = torch.cumprod(alphas, dim=0)
                previous = torch.cat((torch.ones(1, dtype=torch.float64), cumulative[:-1]))
                posterior_variance = betas * (1.0 - previous) / (1.0 - cumulative)
                self.register_buffer("betas", betas.float())
                self.register_buffer("alphas_cumprod", cumulative.float())
                self.register_buffer("sqrt_alphas_cumprod", cumulative.sqrt().float())
                self.register_buffer(
                    "sqrt_one_minus_alphas_cumprod", (1.0 - cumulative).sqrt().float()
                )
                self.register_buffer(
                    "posterior_variance", posterior_variance.clamp(min=1e-20).float()
                )
                self.register_buffer(
                    "posterior_mean_coef1",
                    (betas * previous.sqrt() / (1.0 - cumulative)).float(),
                )
                self.register_buffer(
                    "posterior_mean_coef2",
                    ((1.0 - previous) * alphas.sqrt() / (1.0 - cumulative)).float(),
                )

            def q_sample(self, clean, timestep, noise=None):
                """执行 DDPM 前向扩散，将噪声加入干净影像。"""
                if noise is None:
                    noise = torch.randn_like(clean)
                return (
                    _extract(self.sqrt_alphas_cumprod, timestep, clean.shape) * clean
                    + _extract(
                        self.sqrt_one_minus_alphas_cumprod, timestep, clean.shape
                    )
                    * noise
                )

            def forward(self, image, mask, histogram, timestep=None, noise=None):
                """预测扩散噪声并计算病灶前景归一化损失。"""
                from .losses import masked_foreground_loss

                batch = image.shape[0]
                if timestep is None:
                    timestep = torch.randint(
                        0, self.timesteps, (batch,), device=image.device
                    ).long()
                if noise is None:
                    noise = torch.randn_like(image)
                noisy = self.q_sample(image, timestep, noise)
                predicted_noise = self.denoiser(
                    x=noisy, time=timestep, cond=histogram.to(image.device)
                )
                return masked_foreground_loss(
                    predicted_noise, noise, mask, loss_type=self.loss_type
                )

            def _p_mean_variance(self, noisy, timestep, histogram):
                """由噪声预测计算反向扩散的后验均值和方差。"""
                predicted_noise = self.denoiser(
                    x=noisy, time=timestep, cond=histogram.to(noisy.device)
                )
                alpha = _extract(self.alphas_cumprod, timestep, noisy.shape)
                clean = (
                    noisy
                    - torch.sqrt(torch.clamp(1.0 - alpha, min=1e-12)) * predicted_noise
                ) / torch.sqrt(torch.clamp(alpha, min=1e-12))
                clean = clean.clamp(-1.0, 1.0)
                mean = (
                    _extract(self.posterior_mean_coef1, timestep, noisy.shape) * clean
                    + _extract(self.posterior_mean_coef2, timestep, noisy.shape) * noisy
                )
                variance = _extract(self.posterior_variance, timestep, noisy.shape)
                return mean, variance

            @torch.no_grad()
            def sample_patch(self, background, mask, histogram, *, generator=None):
                """RePaint-style sampling with real background injection at every step."""
                if background.shape != mask.shape:
                    raise ValueError("background and mask shapes differ")
                lesion = mask.to(device=background.device, dtype=torch.bool)
                if not torch.any(lesion):
                    raise ValueError("synthesis mask is empty")
                sample = torch.randn(
                    background.shape,
                    dtype=background.dtype,
                    device=background.device,
                    generator=generator,
                )
                for value in reversed(range(self.timesteps)):
                    timestep = torch.full(
                        (background.shape[0],),
                        value,
                        device=background.device,
                        dtype=torch.long,
                    )
                    background_noise = torch.randn(
                        background.shape,
                        dtype=background.dtype,
                        device=background.device,
                        generator=generator,
                    )
                    noised_background = self.q_sample(
                        background, timestep, background_noise
                    )
                    sample = torch.where(lesion, sample, noised_background)
                    mean, variance = self._p_mean_variance(
                        sample, timestep, histogram
                    )
                    if value > 0:
                        innovation = torch.randn(
                            sample.shape,
                            dtype=sample.dtype,
                            device=sample.device,
                            generator=generator,
                        )
                        sample = mean + variance.sqrt() * innovation
                    else:
                        sample = mean
                return torch.where(lesion, sample, background)

        return _LeFusionH()


def load_model_checkpoint(
    checkpoint_path: str | Path,
    model_config: dict[str, Any],
    device,
    *,
    use_ema: bool = True,
):
    """构建模型并载入普通或 EMA checkpoint 参数。"""
    torch = require_torch()
    model = LeFusionH(model_config).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state = checkpoint.get("ema_model") if use_ema else None
    model.load_state_dict(state or checkpoint["model"])
    model.eval()
    return model, checkpoint
