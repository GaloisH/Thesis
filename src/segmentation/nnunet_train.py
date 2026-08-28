#!/usr/bin/env python3
"""Run nnUNet training from the project YAML configuration."""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "nnUNetseg_train.yaml"


def run_nnunet_with_swanlab() -> None:
    """Enable SwanLab's W&B sync before nnUNet initializes its logger."""
    import swanlab
    from nnunetv2.run.run_training import run_training_entry

    swanlab.sync_wandb()
    run_training_entry()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nnUNet training")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    root_dir = Path(config["root_dir"])
    if not root_dir.is_absolute():
        root_dir = PROJECT_ROOT / root_dir
    results_dir = root_dir / "nnUNet_results" / config["experiment_name"]
    if results_dir.exists():
        raise FileExistsError(
            f"Experiment already exists: {results_dir}. "
            "Change experiment_name to start a new experiment."
        )

    environment = os.environ.copy()
    environment.update(
        {
            "nnUNet_raw": str(root_dir / "nnUNet_raw"),
            "nnUNet_preprocessed": str(root_dir / "nnUNet_preprocessed"),
            "nnUNet_results": str(results_dir),
            "nnUNet_wandb_enabled": "1" if config["wandb"]["enabled"] else "0",
            "nnUNet_wandb_project": str(config["wandb"]["project"]),
        }
    )

    print("Environment")
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        print(f"{name} = {environment[name]}")

    for fold in config["folds"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        environment["WANDB_NAME"] = config["wandb"]["name_template"].format(
            experiment_name=config["experiment_name"],
            task_id=config["task_id"],
            config=config["config"],
            fold=fold,
            timestamp=timestamp,
        )
        print(f"Training fold {fold}: WANDB_NAME={environment['WANDB_NAME']}")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--swanlab-nnunet",
                str(config["task_id"]),
                str(config["config"]),
                str(fold),
                "-tr",
                str(config["trainer"]),
            ],
            check=True,
            env=environment,
        )
        print(f"Fold {fold} finished")

    print("All training finished")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--swanlab-nnunet"]:
        del sys.argv[1]
        run_nnunet_with_swanlab()
    else:
        main()
