#!/usr/bin/env python3
"""nnUNet training launcher — 3d_fullres with 100 epochs.

Replaces train.sh. Reads configuration from config/nnUNetseg_train.yaml.

Usage:
    python src/segmentation/nnunet_train.py
    python src/segmentation/nnunet_train.py --config config/nnUNetseg_train.yaml
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict, project_root: str) -> dict[str, str]:
    root_dir = cfg.get("root_dir", "./datasets")
    if not os.path.isabs(root_dir):
        root_dir = os.path.normpath(os.path.join(project_root, root_dir))
    return {
        "nnUNet_raw": os.path.join(root_dir, "nnUNet_raw"),
        "nnUNet_preprocessed": os.path.join(root_dir, "nnUNet_preprocessed"),
        "nnUNet_results": os.path.join(root_dir, "nnUNet_results"),
    }


def setup_environment(paths: dict[str, str], cfg: dict) -> None:
    for key, value in paths.items():
        os.makedirs(value, exist_ok=True)
        os.environ[key] = value

    wandb_cfg = cfg.get("wandb", {})
    os.environ["nnUNet_wandb_enabled"] = "1" if wandb_cfg.get("enabled", True) else "0"
    os.environ["nnUNet_wandb_project"] = str(wandb_cfg.get("project", "nnUNet_Meningioma"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_prefix = wandb_cfg.get("name_prefix", "Task101_3d_fullres_fold0")
    os.environ["WANDB_NAME"] = f"{name_prefix}_{timestamp}"


def check_preprocessed(preprocessed_dir: str, task_id: str, task_name: str) -> str:
    preprocess_dir = os.path.join(preprocessed_dir, f"Dataset{task_id}_{task_name}")
    if not os.path.isdir(preprocess_dir) or not os.listdir(preprocess_dir):
        print(f"Preprocessed data not found: {preprocess_dir}")
        sys.exit(1)
    return preprocess_dir


def run_command(command: list[str]) -> None:
    print(f"\n>> Running: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        process.wait()
        if process.returncode != 0:
            print(f"Command failed with return code: {process.returncode}")
            sys.exit(process.returncode)
    except Exception as e:
        print(f"Execution error: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="nnUNet 3d_fullres training launcher")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: config/nnUNetseg_train.yaml relative to project root)",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config or os.path.join(project_root, "config", "nnUNetseg_train.yaml")

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)
    paths = resolve_paths(cfg, project_root)
    setup_environment(paths, cfg)

    task_id = str(cfg["task_id"])
    trainer = cfg["trainer"]
    nnunet_config = cfg["config"]
    fold = cfg["fold"]

    print("Environment variables:")
    for key in paths:
        print(f"  {key} = {os.environ[key]}")
    print(f"  WANDB_NAME = {os.environ['WANDB_NAME']}")
    print()

    check_preprocessed(paths["nnUNet_preprocessed"], task_id, cfg.get("task_name", "Meningioma"))

    print("=" * 60)
    print("Starting training")
    print("=" * 60)

    run_command([
        "nnUNetv2_train",
        task_id,
        nnunet_config,
        str(fold),
        "-tr", trainer,
    ])

    print()
    print(f"Training complete. Results saved to: {paths['nnUNet_results']}")


if __name__ == "__main__":
    main()
