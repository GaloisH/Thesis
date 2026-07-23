from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def require_numpy():
    """延迟导入 NumPy，并在缺失时给出可操作的错误。"""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required; install project requirements first") from exc
    return np


def require_nibabel():
    """延迟导入 nibabel，并在缺失时给出可操作的错误。"""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required; install project requirements first") from exc
    return nib


def case_id_from_image(path: str | Path) -> str:
    """从 nnUNet 的影像文件名中提取病例 ID。"""
    name = Path(path).name
    if not name.endswith(".nii.gz"):
        raise ValueError(f"expected .nii.gz image, got {name}")
    stem = name[:-7]
    prefix, separator, channel = stem.rpartition("_")
    if not separator or len(channel) != 4 or not channel.isdigit():
        raise ValueError(f"nnUNet image name must end in _CCCC.nii.gz: {name}")
    return prefix


def image_path(dataset_dir: str | Path, case_id: str, channel: int) -> Path:
    """构造指定病例与通道的 nnUNet 训练影像路径。"""
    return Path(dataset_dir) / "imagesTr" / f"{case_id}_{channel:04d}.nii.gz"


def label_path(dataset_dir: str | Path, case_id: str) -> Path:
    """构造指定病例的 nnUNet 训练标签路径。"""
    return Path(dataset_dir) / "labelsTr" / f"{case_id}.nii.gz"


def discover_cases(dataset_dir: str | Path, channel: int) -> list[str]:
    """扫描指定通道并返回排序后的训练病例 ID。"""
    images = Path(dataset_dir) / "imagesTr"
    return sorted(case_id_from_image(path) for path in images.glob(f"*_{channel:04d}.nii.gz"))


def load_ras(path: str | Path, *, label: bool = False):
    """读取 NIfTI 并转换为 RAS 方向的数组与影像对象。"""
    np = require_numpy()
    nib = require_nibabel()
    source = nib.load(str(path))
    ras = nib.as_closest_canonical(source)
    dtype = np.int16 if label else np.float32
    return np.asarray(ras.dataobj, dtype=dtype), ras


def load_ras_with_source(path: str | Path, *, label: bool = False):
    """同时返回 RAS 数据、RAS 影像和原始方向影像。"""
    np = require_numpy()
    nib = require_nibabel()
    source = nib.load(str(path))
    ras = nib.as_closest_canonical(source)
    dtype = np.int16 if label else np.float32
    return np.asarray(ras.dataobj, dtype=dtype), ras, source


def restore_ras_to_source(data, source):
    """Restore an RAS array to a source image's original voxel orientation."""
    np = require_numpy()
    nib = require_nibabel()
    canonical = nib.orientations.axcodes2ornt(("R", "A", "S"))
    source_orientation = nib.orientations.io_orientation(source.affine)
    transform = nib.orientations.ornt_transform(canonical, source_orientation)
    restored = nib.orientations.apply_orientation(np.asarray(data), transform)
    if tuple(restored.shape) != tuple(source.shape):
        raise ValueError(
            f"restored shape {restored.shape} does not match source {source.shape}"
        )
    return restored


def save_like(data, reference, path: str | Path, *, dtype=None) -> None:
    """使用参考影像的 affine 与 header 保存 NIfTI 数据。"""
    np = require_numpy()
    nib = require_nibabel()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(data, dtype=dtype)
    header = reference.header.copy()
    if dtype is not None:
        header.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(array, reference.affine, header), str(target))


def write_json(path: str | Path, value: Any) -> None:
    """以 UTF-8 和缩进格式写入 JSON 文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    """读取 UTF-8 JSON 文件并返回反序列化对象。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_hash(value: Any) -> str:
    """计算与字典键顺序无关的稳定 SHA-256 摘要。"""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
