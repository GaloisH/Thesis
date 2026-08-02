from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..synthesis.runtime import load_inference_runtime
from .inputs import batch_entries, case_id_from_path
from .pipeline import _process_case
from .reporting import write_case_index


def _validate_visualization_config(config: dict[str, Any]) -> None:
    cfg = config["visualization"]
    if str(cfg.get("format", "png")).lower() != "png":
        raise ValueError("visualization.format currently supports only png")
    if int(cfg["dpi"]) <= 0:
        raise ValueError("visualization.dpi must be positive")
    if not 0.0 <= float(cfg["mask_alpha"]) <= 1.0:
        raise ValueError("visualization.mask_alpha must be between 0 and 1")
    percentiles = cfg["intensity_percentiles"]
    if (
        len(percentiles) != 2
        or not 0 <= float(percentiles[0]) < float(percentiles[1]) <= 100
    ):
        raise ValueError(
            "visualization.intensity_percentiles must be two increasing values in [0, 100]"
        )


def visualize(
    config: dict[str, Any],
    *,
    image: str | Path | None = None,
    mask: str | Path | None = None,
    output_dir: str | Path | None = None,
    case_id: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run one fixed-mask case or all train and validation cases by default."""
    if bool(image) != bool(mask):
        raise ValueError("--image and --mask must be provided together")
    _validate_visualization_config(config)
    base_seed = int(config["seed"] if seed is None else seed)
    configured_output = Path(config["visualization"]["output_dir"])
    requested_output = Path(output_dir) if output_dir else configured_output
    single_case = bool(image)
    if single_case:
        resolved_case = case_id or case_id_from_path(image)
        entries = [{"case_id": resolved_case, "image": Path(image), "mask": Path(mask)}]
        root_output = requested_output if output_dir else requested_output / resolved_case
    else:
        if case_id:
            raise ValueError("--case-id is only valid together with --image and --mask")
        entries = batch_entries(config)
        root_output = requested_output
    for entry in entries:
        if not entry["case_id"]:
            raise ValueError("case_id must not be empty")
        if not entry["image"].is_file():
            raise FileNotFoundError(f"image not found: {entry['image']}")
        if not entry["mask"].is_file():
            raise FileNotFoundError(f"mask not found: {entry['mask']}")

    model, checkpoint, device, histogram_library = load_inference_runtime(config)
    records = []
    for index, entry in enumerate(tqdm(entries, desc="Visualizing", unit="case")):
        target = root_output if single_case else root_output / entry["case_id"]
        records.append(
            _process_case(
                config,
                model,
                checkpoint,
                device,
                image_path=entry["image"],
                mask_path=entry["mask"],
                output_dir=target,
                case_id=entry["case_id"],
                seed=base_seed + index,
                histogram_library=histogram_library,
                prepared_entry=entry.get("prepared_entry"),
                prepared_dir=(
                    Path(config["data"]["prepared_dir"])
                    if entry.get("prepared_entry") is not None
                    else None
                ),
            )
        )
    if not single_case:
        root_output.mkdir(parents=True, exist_ok=True)
        write_case_index(root_output, records)
    return {
        "cases": len(records),
        "qc_passed": sum(bool(record["qc"]["passed"]) for record in records),
        "output_dir": str(root_output),
        "records": [record["case_id"] for record in records],
    }
