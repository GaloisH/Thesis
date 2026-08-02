from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ._checkpoints import restore_training_checkpoint, save_training_checkpoint
from .data import MeningitisPatchDataset
from .io import read_json, write_json
from .logger import get_logger
from .model import LeFusionH, require_torch

logger = get_logger(__name__)


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
    """Run training loop with gradient accumulation, AMP, EMA, early stopping, and checkpointing."""
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

    logger.info("=== Starting training ===")
    logger.info("Output directory: %s", output_dir)
    logger.info("Device: %s", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Max steps: %d", train_cfg["max_steps"])

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
    logger.info(
        "Data: train=%d, val=%d, batch_size=%d, accumulation=%d, effective_batch=%d",
        len(train_dataset), len(val_dataset), batch_size, accumulation,
        batch_size * accumulation,
    )

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
        logger.info("Resuming from checkpoint: %s", resume)
        restored = restore_training_checkpoint(
            resume,
            model=model,
            ema_model=ema.model,
            optimizer=optimizer,
            scaler=scaler,
        )
        global_step = restored["global_step"]
        best_loss = restored["best_val_loss"]
        history = restored["history"]
        logger.info("Resumed at step %d, best_val_loss=%.6f", global_step, best_loss)

    def save_checkpoint(name: str) -> Path:
        return save_training_checkpoint(
            output_dir / name,
            model=model,
            ema_model=ema.model,
            optimizer=optimizer,
            scaler=scaler,
            global_step=global_step,
            best_val_loss=best_loss,
            history=history,
            config=config,
            manifest_hash=manifest["hash"],
            split_hash=split["hash"],
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    max_steps = int(train_cfg["max_steps"])
    validate_every = int(train_cfg["validate_every"])
    save_every = int(train_cfg["save_every"])
    patience = int(train_cfg["patience_validations"])
    pbar = tqdm(total=max_steps, desc="Training", unit="step", initial=global_step)
    while global_step < max_steps:
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
            pbar.update(1)
            pbar.set_postfix(loss=f"{float(loss.item() * accumulation):.4f}")

            if global_step % validate_every == 0:
                val_loss = _validate(ema.model, val_loader, device)
                record = {
                    "step": global_step,
                    "train_loss": float(loss.item() * accumulation),
                    "val_loss": val_loss,
                }
                history.append(record)
                write_json(output_dir / "history.json", history)
                pbar.set_postfix(train_loss=record["train_loss"], val_loss=f"{val_loss:.4f}")
                logger.info(
                    "Step %d: train_loss=%.6f, val_loss=%.6f",
                    global_step, record["train_loss"], val_loss,
                )
                if val_loss < best_loss:
                    best_loss = val_loss
                    stale_validations = 0
                    save_checkpoint("best.pt")
                    logger.info("New best model at step %d (val_loss=%.6f)", global_step, best_loss)
                else:
                    stale_validations += 1
                if stale_validations >= patience:
                    pbar.close()
                    save_checkpoint("last.pt")
                    logger.info(
                        "Early stopping at step %d (no improvement for %d validations)",
                        global_step, stale_validations,
                    )
                    return {
                        "status": "early_stopped",
                        "global_step": global_step,
                        "best_val_loss": best_loss,
                        "output_dir": str(output_dir),
                    }
            if global_step % save_every == 0:
                path = save_checkpoint(f"step_{global_step:06d}.pt")
                logger.info("Checkpoint saved: %s", path.name)
            if global_step >= max_steps:
                break
    pbar.close()
    save_checkpoint("last.pt")
    if not (output_dir / "best.pt").exists():
        save_checkpoint("best.pt")
    logger.info("=== Training complete: step=%d, best_val_loss=%.6f ===", global_step, best_loss)
    return {
        "status": "completed",
        "global_step": global_step,
        "best_val_loss": best_loss,
        "output_dir": str(output_dir),
    }


def validate(config: dict[str, Any]) -> dict[str, Any]:
    """Load best checkpoint and produce a standalone validation loss report."""
    torch = require_torch()
    from torch.utils.data import DataLoader

    prepared_dir = Path(config["data"]["prepared_dir"])
    checkpoint_path = config["synthesis"]["checkpoint"]
    logger.info("=== Starting validation ===")
    logger.info("Checkpoint: %s", checkpoint_path)
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
        checkpoint_path, config["model"], device
    )
    loss = _validate(model, loader, device)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("global_step", -1)),
        "validation_patches": len(dataset),
        "foreground_noise_loss": loss,
        "manifest_hash": read_json(prepared_dir / "manifest.json")["hash"],
    }
    write_json(Path(config["training"]["output_dir"]) / "validation.json", report)
    logger.info("Validation loss: %.6f (%d patches)", loss, len(dataset))
    return report
