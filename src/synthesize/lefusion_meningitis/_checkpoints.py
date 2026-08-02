from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .model import require_torch


def restore_training_checkpoint(
    path: str | Path,
    *,
    model,
    ema_model,
    optimizer,
    scaler,
) -> dict[str, Any]:
    """Restore trainable state and return counters and history to the caller."""
    torch = require_torch()
    checkpoint = torch.load(str(path), map_location=next(model.parameters()).device)
    model.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema_model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler_state = checkpoint.get("scaler")
    if scaler_state:
        scaler.load_state_dict(scaler_state)
    rng = checkpoint.get("rng") or {}
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        import numpy as np

        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    return {
        "global_step": int(checkpoint["global_step"]),
        "best_val_loss": float(checkpoint.get("best_val_loss", float("inf"))),
        "history": list(checkpoint.get("history", [])),
    }


def save_training_checkpoint(
    target: str | Path,
    *,
    model,
    ema_model,
    optimizer,
    scaler,
    global_step: int,
    best_val_loss: float,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    manifest_hash: str,
    split_hash: str,
) -> Path:
    """Save model state, provenance, and random-number generator states."""
    import numpy as np

    torch = require_torch()
    target = Path(target)
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": int(global_step),
            "best_val_loss": float(best_val_loss),
            "history": history,
            "config": config,
            "manifest_hash": manifest_hash,
            "split_hash": split_hash,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
            },
        },
        target,
    )
    return target
