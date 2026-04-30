#!/usr/bin/env python3
"""nnUNet plan & preprocess launcher.

Replaces preprocess.sh. Reads configuration from config/nnUNetseg_preprocess.yaml.

Usage:
    python src/segmentation/nnunet_preprocess.py
    python src/segmentation/nnunet_preprocess.py --config config/nnUNetseg_preprocess.yaml
"""

import argparse
import os
import subprocess
import sys
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


def setup_environment(paths: dict[str, str]) -> None:
    for key, value in paths.items():
        os.makedirs(value, exist_ok=True)
        os.environ[key] = value


def check_raw_data(raw_dir: str, task_id: str, task_name: str) -> str:
    task_dir = os.path.join(raw_dir, f"Dataset{task_id}_{task_name}")
    if not os.path.isdir(task_dir) or not os.listdir(task_dir):
        print(f"No data detected at: {task_dir}")
        print("Please run: python src/segmentation/prepare_data.py")
        sys.exit(1)
    return task_dir


def run_command(command: list[str]) -> None:
    print(f">> Running: {' '.join(command)}")
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
    parser = argparse.ArgumentParser(description="nnUNet plan & preprocess launcher")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: config/nnUNetseg_preprocess.yaml relative to project root)",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config or os.path.join(project_root, "config", "nnUNetseg_preprocess.yaml")

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)
    paths = resolve_paths(cfg, project_root)
    setup_environment(paths)

    task_id = str(cfg["task_id"])
    task_name = cfg.get("task_name", "Meningioma")
    nnunet_config = cfg["config"]

    print("Environment variables:")
    for key in paths:
        print(f"  {key} = {os.environ[key]}")
    print()

    check_raw_data(paths["nnUNet_raw"], task_id, task_name)

    print("=" * 60)
    print("Plan & Preprocess")
    print("=" * 60)

    command = [
        "nnUNetv2_plan_and_preprocess",
        "-d", task_id,
        "-c", nnunet_config,
    ]
    if cfg.get("verify_dataset_integrity", True):
        command.append("--verify_dataset_integrity")

    run_command(command)

    print()
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
