from __future__ import annotations

import argparse
import json
from pathlib import Path


def visualize_loss(loss_json: str | Path) -> Path:
    """Plot training and validation losses from a history JSON file."""
    history_path = Path(loss_json)
    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)

    if not isinstance(history, list) or not history:
        raise ValueError("loss JSON must be a non-empty list")

    required_keys = {"step", "train_loss", "val_loss"}
    if any(not isinstance(record, dict) or not required_keys <= record.keys() for record in history):
        raise ValueError("each loss record must contain step, train_loss, and val_loss")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    steps = [record["step"] for record in history]
    train_losses = [record["train_loss"] for record in history]
    val_losses = [record["val_loss"] for record in history]

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(x=steps, y=train_losses, marker="o", ax=axes[0])
    sns.lineplot(x=steps, y=val_losses, marker="o", ax=axes[1])

    axes[0].set(title="Training Loss", xlabel="Step", ylabel="Loss")
    axes[1].set(title="Validation Loss", xlabel="Step", ylabel="Loss")
    figure.tight_layout()

    output_dir = Path(__file__).parent / "figs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "loss_curves.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LeFusion training losses")
    parser.add_argument("loss_json", type=Path, help="path to history.json")
    args = parser.parse_args()
    print(visualize_loss(args.loss_json))


if __name__ == "__main__":
    main()
