from __future__ import annotations

from .logger import get_logger

logger = get_logger(__name__)


def masked_foreground_loss(prediction, target, mask, loss_type: str = "l1", eps: float = 1e-8):
    """Compute foreground-normalized noise loss, independently per batch."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for LeFusion training") from exc

    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if mask.ndim != prediction.ndim:
        raise ValueError("mask must have the same rank as prediction")
    if mask.shape[0] != prediction.shape[0] or mask.shape[2:] != prediction.shape[2:]:
        raise ValueError("mask batch/spatial shape differs from prediction")
    if mask.shape[1] == 1 and prediction.shape[1] != 1:
        mask = mask.expand(-1, prediction.shape[1], *([-1] * (prediction.ndim - 2)))
    if mask.shape != prediction.shape:
        raise ValueError("mask channel shape is incompatible with prediction")

    difference = prediction - target
    if loss_type == "l1":
        error = difference.abs()
    elif loss_type == "l2":
        error = difference.square()
    else:
        raise ValueError(f"unsupported loss_type: {loss_type}")

    weights = mask.to(dtype=error.dtype)
    reduce_dims = tuple(range(1, error.ndim))
    numerator = (error * weights).sum(dim=reduce_dims)
    foreground = weights.sum(dim=reduce_dims)
    if torch.any(foreground <= 0):
        raise ValueError("foreground mask is empty")
    denominator = foreground.clamp_min(eps)
    return (numerator / denominator).mean()
