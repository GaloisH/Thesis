from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import torch

from ..model import load_model_checkpoint


def load_inference_runtime(config: dict[str, Any]):
    """加载训练好的模型、设备和训练直方图库以进行推理。"""
    histogram_path = Path(config["data"]["prepared_dir"]) / "train_histograms.npy"
    if not histogram_path.is_file():
        raise FileNotFoundError(f"training histogram library not found: {histogram_path}")
    histograms = np.load(histogram_path).copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_checkpoint(
        config["synthesis"]["checkpoint"], config["model"], device
    )
    return model, checkpoint, device, histograms
