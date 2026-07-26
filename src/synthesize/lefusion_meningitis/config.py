from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置字典，并让命令行覆盖值优先。"""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load YAML config, apply overrides, and resolve all project-relative paths."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install project requirements first") from exc

    config_path = Path(path).resolve()
    logger.info("Loading config from %s", config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if overrides:
        logger.info("Applying %d config override(s)", len(overrides))
        config = _deep_merge(config, overrides)

    declared_root = Path(config.get("project_root", "."))
    root = declared_root if declared_root.is_absolute() else PROJECT_ROOT / declared_root
    config["project_root"] = str(root.resolve())
    logger.info("Project root: %s", config["project_root"])

    for section, keys in {
        "data": ("source_dataset", "registered_dir", "prepared_dir"),
        "training": ("output_dir", "resume"),
        "synthesis": ("checkpoint", "output_dir"),
        "visualization": ("output_dir",),
        "export": ("output_dataset",),
        "evaluation": ("output", "predictions_dir", "references_dir"),
    }.items():
        values = config.get(section, {})
        for key in keys:
            value = values.get(key)
            if value in (None, ""):
                continue
            candidate = Path(value)
            values[key] = str(candidate if candidate.is_absolute() else root / candidate)
    return config


def require_keys(config: dict[str, Any], *paths: str) -> None:
    """检查点分隔的必需配置键是否全部存在。"""
    for dotted in paths:
        value: Any = config
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f"missing configuration key: {dotted}")
            value = value[part]
