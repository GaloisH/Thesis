from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import load_ras, read_json, require_numpy, write_json
from .logger import get_logger

logger = get_logger(__name__)


def segmentation_metrics(prediction, reference, spacing, tolerance_mm: float = 1.0):
    """计算 Dice、HD95、NSD、病灶召回率和假阳性数。"""
    np = require_numpy()
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt, label
    except ImportError as exc:
        raise RuntimeError("SciPy is required for evaluation") from exc

    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    intersection = int(np.count_nonzero(prediction & reference))
    denominator = int(prediction.sum() + reference.sum())
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator

    pred_surface = prediction & ~binary_erosion(prediction)
    ref_surface = reference & ~binary_erosion(reference)
    if pred_surface.any() and ref_surface.any():
        distance_to_ref = distance_transform_edt(~ref_surface, sampling=spacing)
        distance_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
        distances = np.concatenate(
            (distance_to_ref[pred_surface], distance_to_pred[ref_surface])
        )
        hd95 = float(np.percentile(distances, 95))
        surface_hits = int(
            np.count_nonzero(distance_to_ref[pred_surface] <= tolerance_mm)
            + np.count_nonzero(distance_to_pred[ref_surface] <= tolerance_mm)
        )
        nsd = surface_hits / max(int(pred_surface.sum() + ref_surface.sum()), 1)
    elif not pred_surface.any() and not ref_surface.any():
        hd95, nsd = 0.0, 1.0
    else:
        hd95, nsd = float("inf"), 0.0

    ref_components, ref_count = label(reference)
    detected = sum(
        bool(np.any(prediction[ref_components == component]))
        for component in range(1, ref_count + 1)
    )
    pred_components, pred_count = label(prediction)
    false_positives = sum(
        not bool(np.any(reference[pred_components == component]))
        for component in range(1, pred_count + 1)
    )
    return {
        "dice": float(dice),
        "hd95": hd95,
        "nsd": float(nsd),
        "lesion_recall": 1.0 if ref_count == 0 else detected / ref_count,
        "false_positive_lesions": int(false_positives),
        "reference_lesions": int(ref_count),
        "reference_voxels": int(reference.sum()),
    }


def _volume_stratum(voxels: int) -> str:
    """按参考病灶体素数划分小、中、大病例层级。"""
    if voxels < 100:
        return "small"
    if voxels < 1000:
        return "medium"
    return "large"


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    """Aggregate synthesis QC and optionally evaluate a predictions directory."""
    np = require_numpy()
    eval_cfg = config["evaluation"]
    synthesis_dir = Path(config["synthesis"]["output_dir"])
    logger.info("=== Starting evaluation ===")
    logger.info("Synthesis directory: %s", synthesis_dir)

    metadata_paths = sorted((synthesis_dir / "metadata").glob("*.json"))
    metadata = [read_json(path) for path in metadata_paths]
    logger.info("Loaded %d synthesis metadata records", len(metadata))

    report: dict[str, Any] = {
        "synthesis": {
            "accepted": len(metadata),
            "qc_rate": (
                read_json(synthesis_dir / "summary.json").get("qc_rate", 0.0)
                if (synthesis_dir / "summary.json").exists()
                else None
            ),
            "background_exact_rate": (
                float(np.mean([item["qc"]["background_exact"] for item in metadata]))
                if metadata
                else 0.0
            ),
            "mean_boundary_jump": (
                float(np.mean([item["qc"]["boundary_jump"] for item in metadata]))
                if metadata
                else None
            ),
        }
    }

    predictions_dir = eval_cfg.get("predictions_dir")
    references_dir = eval_cfg.get("references_dir")
    if predictions_dir and references_dir:
        logger.info("Computing segmentation metrics: preds=%s, refs=%s", predictions_dir, references_dir)
        cases: list[dict[str, Any]] = []
        prediction_paths = sorted(Path(predictions_dir).glob("*.nii.gz"))
        for prediction_path in prediction_paths:
            reference_path = Path(references_dir) / prediction_path.name
            if not reference_path.exists():
                continue
            prediction, prediction_image = load_ras(prediction_path, label=True)
            reference, reference_image = load_ras(reference_path, label=True)
            if prediction.shape != reference.shape:
                raise ValueError(f"shape mismatch for {prediction_path.name}")
            spacing = tuple(float(value) for value in reference_image.header.get_zooms()[:3])
            values = segmentation_metrics(
                prediction > 0,
                reference > 0,
                spacing,
                float(eval_cfg["nsd_tolerance_mm"]),
            )
            values["case_id"] = prediction_path.name[:-7]
            values["volume_stratum"] = _volume_stratum(values["reference_voxels"])
            cases.append(values)
        finite_hd95 = [item["hd95"] for item in cases if np.isfinite(item["hd95"])]
        report["segmentation"] = {
            "cases": cases,
            "mean": {
                "dice": float(np.mean([item["dice"] for item in cases])) if cases else None,
                "hd95": float(np.mean(finite_hd95)) if finite_hd95 else None,
                "nsd": float(np.mean([item["nsd"] for item in cases])) if cases else None,
                "lesion_recall": (
                    float(np.mean([item["lesion_recall"] for item in cases]))
                    if cases
                    else None
                ),
                "false_positive_lesions_per_case": (
                    float(np.mean([item["false_positive_lesions"] for item in cases]))
                    if cases
                    else None
                ),
            },
        }
        logger.info(
            "Segmentation: %d cases, mean Dice=%.4f, mean HD95=%.2f",
            len(cases),
            report["segmentation"]["mean"]["dice"] or 0.0,
            report["segmentation"]["mean"]["hd95"] or 0.0,
        )
    write_json(eval_cfg["output"], report)
    logger.info("Evaluation report saved to %s", eval_cfg["output"])
    return report
