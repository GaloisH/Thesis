from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any

from .data import MeningitisPatchDataset
from .io import read_json, stable_hash, write_json
from .model import LeFusionH, require_torch


class EMA:
    """维护用于稳定验证与采样的指数滑动平均模型。"""

    def __init__(self, model, decay: float):
        """复制初始模型并设置 EMA 衰减率。"""
        self.decay = float(decay)
        self.model = deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def update(self, model) -> None:
        """用当前训练参数原地更新 EMA 参数。"""
        with require_torch().no_grad():
            source = model.state_dict()
            for name, value in self.model.state_dict().items():
                if value.dtype.is_floating_point:
                    value.lerp_(source[name], 1.0 - self.decay)
                else:
                    value.copy_(source[name])


def _seed_everything(seed: int) -> None:
    """固定 Python、NumPy、CPU 和 CUDA 随机种子。"""
    import numpy as np

    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move(batch, device):
    """将一个训练 batch 的三个张量移动到目标设备。"""
    return (
        batch["image"].to(device, non_blocking=True),
        batch["mask"].to(device, non_blocking=True),
        batch["histogram"].to(device, non_blocking=True),
    )


def _validate(model, loader, device) -> float:
    """计算整个验证集的平均病灶聚焦损失。"""
    torch = require_torch()
    model.eval()
    total = 0.0
    samples = 0
    with torch.no_grad():
        for batch in loader:
            image, mask, histogram = _move(batch, device)
            loss = model(image, mask, histogram)
            total += float(loss.item()) * image.shape[0]
            samples += image.shape[0]
    model.train()
    return total / max(samples, 1)


def train(config: dict[str, Any]) -> dict[str, Any]:
    """运行带梯度累积、AMP、EMA、早停和断点的训练循环。"""
    torch = require_torch()
    from torch.utils.data import DataLoader

    seed = int(config["seed"])
    _seed_everything(seed)
    data_cfg = config["data"]
    train_cfg = config["training"]
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = Path(data_cfg["prepared_dir"])
    manifest = read_json(prepared_dir / "manifest.json")
    split = read_json(prepared_dir / "split.json")

    train_dataset = MeningitisPatchDataset(
        prepared_dir,
        "train",
        augmentation=config.get("augmentation"),
        seed=seed,
    )
    val_dataset = MeningitisPatchDataset(prepared_dir, "val", seed=seed)
    batch_size = int(train_cfg["batch_size"])
    accumulation = max(
        1, int(train_cfg["effective_batch_size"]) // max(batch_size, 1)
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": int(train_cfg["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeFusionH(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=1e-4
    )
    amp_enabled = bool(train_cfg["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    ema = EMA(model, float(train_cfg["ema_decay"]))

    global_step = 0
    best_loss = float("inf")
    stale_validations = 0
    history: list[dict[str, Any]] = []
    resume = train_cfg.get("resume")
    if resume:
        checkpoint = torch.load(str(resume), map_location=device)
        model.load_state_dict(checkpoint["model"])
        ema.model.load_state_dict(checkpoint["ema_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        global_step = int(checkpoint["global_step"])
        best_loss = float(checkpoint.get("best_val_loss", best_loss))
        history = list(checkpoint.get("history", []))

    def save_checkpoint(name: str) -> Path:
        """保存模型、优化器、配置、数据哈希和随机状态。"""
        target = output_dir / name
        torch.save(
            {
                "model": model.state_dict(),
                "ema_model": ema.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "global_step": global_step,
                "best_val_loss": best_loss,
                "history": history,
                "config": config,
                "manifest_hash": manifest["hash"],
                "split_hash": split["hash"],
                "rng": {
                    "python": random.getstate(),
                    "numpy": __import__("numpy").random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            },
            target,
        )
        return target

    model.train()
    optimizer.zero_grad(set_to_none=True)
    while global_step < int(train_cfg["max_steps"]):
        for batch in train_loader:
            image, mask, histogram = _move(batch, device)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss = model(image, mask, histogram) / accumulation
            scaler.scale(loss).backward()
            if (global_step + 1) % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            global_step += 1

            if global_step % int(train_cfg["validate_every"]) == 0:
                val_loss = _validate(ema.model, val_loader, device)
                record = {
                    "step": global_step,
                    "train_loss": float(loss.item() * accumulation),
                    "val_loss": val_loss,
                }
                history.append(record)
                write_json(output_dir / "history.json", history)
                if val_loss < best_loss:
                    best_loss = val_loss
                    stale_validations = 0
                    save_checkpoint("best.pt")
                else:
                    stale_validations += 1
                if stale_validations >= int(train_cfg["patience_validations"]):
                    save_checkpoint("last.pt")
                    return {
                        "status": "early_stopped",
                        "global_step": global_step,
                        "best_val_loss": best_loss,
                        "output_dir": str(output_dir),
                    }
            if global_step % int(train_cfg["save_every"]) == 0:
                save_checkpoint(f"step_{global_step:06d}.pt")
            if global_step >= int(train_cfg["max_steps"]):
                break
    save_checkpoint("last.pt")
    if not (output_dir / "best.pt").exists():
        save_checkpoint("best.pt")
    return {
        "status": "completed",
        "global_step": global_step,
        "best_val_loss": best_loss,
        "output_dir": str(output_dir),
    }


def validate(config: dict[str, Any]) -> dict[str, Any]:
    """加载最佳 checkpoint 并生成独立验证损失报告。"""
    torch = require_torch()
    from torch.utils.data import DataLoader

    prepared_dir = Path(config["data"]["prepared_dir"])
    dataset = MeningitisPatchDataset(prepared_dir, "val", seed=int(config["seed"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from .model import load_model_checkpoint

    model, checkpoint = load_model_checkpoint(
        config["synthesis"]["checkpoint"], config["model"], device
    )
    loss = _validate(model, loader, device)
    report = {
        "checkpoint": str(config["synthesis"]["checkpoint"]),
        "checkpoint_step": int(checkpoint.get("global_step", -1)),
        "validation_patches": len(dataset),
        "foreground_noise_loss": loss,
        "manifest_hash": read_json(prepared_dir / "manifest.json")["hash"],
    }
    write_json(Path(config["training"]["output_dir"]) / "validation.json", report)
    return report
