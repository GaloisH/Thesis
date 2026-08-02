from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .io import image_path, label_path, read_json, stable_hash, write_json
from .logger import get_logger

logger = get_logger(__name__)


def _copy(source: Path, destination: Path) -> None:
    """保留文件元数据地复制一个数据集文件。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def export_nnunet(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """Export real + QC-passed synthetic cases as a standalone nnUNet dataset."""
    export_cfg = config["export"]
    data_cfg = config["data"]
    synthetic_ratio = float(export_cfg["synthetic_ratio"])
    if not 0.0 <= synthetic_ratio <= 1.0:
        raise ValueError(
            "export.synthetic_ratio must be between 0 and 1 because synthesis "
            "produces at most one cumulative output per source case"
        )
    output = Path(export_cfg["output_dataset"])
    logger.info("=== Starting nnUNet export ===")
    logger.info("Output dataset: %s", output)

    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(
                f"output dataset is not empty: {output}; choose a new Dataset ID or use --force"
            )
        resolved = output.resolve()
        project_root = Path(config["project_root"]).resolve()
        if project_root not in resolved.parents or resolved == project_root:
            raise ValueError(f"refusing to replace unsafe output path: {resolved}")
        logger.warning("Removing existing output dataset: %s", resolved)
        shutil.rmtree(resolved)

    for name in ("imagesTr", "labelsTr", "imagesTs", "labelsTs", "metadata"):
        (output / name).mkdir(parents=True, exist_ok=True)

    prepared = Path(data_cfg["prepared_dir"])
    split = read_json(prepared / "split.json")
    source_dataset = Path(data_cfg["source_dataset"])
    channel = int(data_cfg["channel"])
    training_cases: list[str] = []

    if bool(export_cfg.get("include_real", True)):
        logger.info("Copying %d real training cases", len(split["cases"]["train"]))
        for case_id in tqdm(split["cases"]["train"], desc="Exporting real cases", unit="case"):
            _copy(
                image_path(source_dataset, case_id, channel),
                output / "imagesTr" / f"{case_id}_0000.nii.gz",
            )
            _copy(label_path(source_dataset, case_id), output / "labelsTr" / f"{case_id}.nii.gz")
            training_cases.append(case_id)

    synthesis_dir = Path(config["synthesis"]["output_dir"])
    synthesis_summary_path = synthesis_dir / "summary.json"
    if synthesis_summary_path.is_file():
        synthesis_summary = read_json(synthesis_summary_path)
        metadata_files = [
            synthesis_dir / "metadata" / f"{sample_id}.json"
            for sample_id in synthesis_summary.get("records", [])
        ]
    else:
        metadata_files = sorted((synthesis_dir / "metadata").glob("*.json"))
    metadata_files = [path for path in metadata_files if path.is_file()]
    target_synthetic = round(len(split["cases"]["train"]) * synthetic_ratio)
    selected = metadata_files[:target_synthetic]
    if len(selected) < target_synthetic:
        raise RuntimeError(
            f"requested {target_synthetic} synthetic cases, only {len(selected)} passed QC"
        )
    logger.info(
        "Copying %d synthetic cases (ratio=%.2f)", len(selected), export_cfg["synthetic_ratio"]
    )
    synthetic_cases: list[str] = []
    for metadata_path in tqdm(selected, desc="Exporting synthetic cases", unit="case"):
        metadata = read_json(metadata_path)
        sample_id = metadata["sample_id"]
        _copy(Path(metadata["outputs"]["image"]), output / "imagesTr" / f"{sample_id}_0000.nii.gz")
        _copy(Path(metadata["outputs"]["label"]), output / "labelsTr" / f"{sample_id}.nii.gz")
        _copy(metadata_path, output / "metadata" / metadata_path.name)
        synthetic_cases.append(sample_id)

    logger.info("Copying %d test cases", len(split["cases"]["test"]))
    for case_id in tqdm(split["cases"]["test"], desc="Exporting test cases", unit="case"):
        _copy(
            image_path(source_dataset, case_id, channel),
            output / "imagesTs" / f"{case_id}_0000.nii.gz",
        )
        _copy(label_path(source_dataset, case_id), output / "labelsTs" / f"{case_id}.nii.gz")

    dataset_json = {
        "channel_names": {"0": "channel_0"},
        "labels": {"background": 0, "meningitis": int(data_cfg["label_id"])},
        "numTraining": len(training_cases) + len(synthetic_cases),
        "file_ending": ".nii.gz",
    }
    write_json(output / "dataset.json", dataset_json)
    provenance = {
        "source_dataset": str(source_dataset),
        "source_split_hash": split["hash"],
        "channel": channel,
        "real_training_cases": training_cases,
        "synthetic_cases": synthetic_cases,
        "held_out_test_cases": split["cases"]["test"],
        "synthetic_ratio": synthetic_ratio,
        "single_channel_mvp": True,
    }
    provenance["hash"] = stable_hash(provenance)
    write_json(output / "metadata" / "provenance.json", provenance)
    logger.info(
        "=== Export complete: real=%d, synthetic=%d, test=%d, numTraining=%d ===",
        len(training_cases), len(synthetic_cases),
        len(split["cases"]["test"]), dataset_json["numTraining"],
    )
    return {
        "output_dataset": str(output),
        "real_training": len(training_cases),
        "synthetic_training": len(synthetic_cases),
        "test": len(split["cases"]["test"]),
        "numTraining": dataset_json["numTraining"],
    }
