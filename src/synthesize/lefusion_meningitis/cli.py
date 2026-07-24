from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .logger import get_logger, setup_logging

logger = get_logger(__name__)


DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "lefusion_meningitis.yaml"


def _parse_value(value: str) -> Any:
    """使用 YAML 语义解析命令行中的标量或列表值。"""
    try:
        import yaml
    except ImportError:
        return value
    return yaml.safe_load(value)


def _overrides(values: list[str]) -> dict[str, Any]:
    """将重复的 key=value 参数转换为嵌套配置字典。"""
    result: dict[str, Any] = {}
    for expression in values:
        if "=" not in expression:
            raise ValueError(f"override must use key=value: {expression}")
        dotted, raw = expression.split("=", 1)
        target = result
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = _parse_value(raw)
    return result


def build_parser() -> argparse.ArgumentParser:
    """构建包含六个流水线子命令的命令行解析器。"""
    parser = argparse.ArgumentParser(description="Meningitis LeFusion-H pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a nested YAML value; may be repeated",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="build split, patches, histograms and position prior")
    commands.add_parser("train", help="train the lesion-focused diffusion model")
    commands.add_parser("validate", help="evaluate foreground noise loss on validation patches")
    commands.add_parser("synthesize", help="generate full-volume synthetic training pairs")
    export_parser = commands.add_parser("export", help="export an isolated nnUNet dataset")
    export_parser.add_argument(
        "--force", action="store_true", help="replace only the configured output dataset"
    )
    commands.add_parser("evaluate", help="aggregate synthesis and segmentation metrics")
    return parser


def main() -> None:
    """Load config and dispatch CLI sub-commands with logging."""
    args = build_parser().parse_args()
    config = load_config(args.config, _overrides(args.set))

    log_level = config.get("logging", {}).get("level", "INFO")
    log_file = config.get("logging", {}).get("file")
    setup_logging(level=log_level, log_file=log_file)

    logger.info("Dispatching command: %s", args.command)
    logger.info("Config: %s", args.config)

    if args.command == "prepare":
        from .data import prepare

        result = prepare(config)
    elif args.command == "train":
        from .training import train

        result = train(config)
    elif args.command == "validate":
        from .training import validate

        result = validate(config)
    elif args.command == "synthesize":
        from .synthesis import synthesize

        result = synthesize(config)
    elif args.command == "export":
        from .export import export_nnunet

        result = export_nnunet(config, force=args.force)
    elif args.command == "evaluate":
        from .evaluation import evaluate

        result = evaluate(config)
    else:
        raise AssertionError(args.command)
    logger.info("Command %s completed", args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
