#!/usr/bin/env python3
"""Run nnUNet planning and preprocessing from the project YAML configuration."""

import argparse
import os
import subprocess
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "nnUNetseg_preprocess.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nnUNet planning and preprocessing")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    root_dir = Path(config["root_dir"])
    if not root_dir.is_absolute():
        root_dir = PROJECT_ROOT / root_dir

    environment = os.environ.copy()
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        path = root_dir / name
        path.mkdir(parents=True, exist_ok=True)
        environment[name] = str(path)
        print(f"{name} = {path}")

    command = [
        "nnUNetv2_plan_and_preprocess",
        "-d",
        str(config["task_id"]),
        "-c",
        str(config["config"]),
    ]
    if config["verify_dataset_integrity"]:
        command.append("--verify_dataset_integrity")

    subprocess.run(command, check=True, env=environment)
    print("Preprocessing finished")


if __name__ == "__main__":
    main()
