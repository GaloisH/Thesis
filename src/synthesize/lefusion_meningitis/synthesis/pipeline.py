from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..data import denormalize_image, robust_normalize
from ..io import (
    image_path,
    label_path,
    load_ras_with_source,
    read_json,
    require_numpy,
    restore_ras_to_source,
    save_like,
    stable_hash,
    write_json,
)
from ..logger import get_logger
from .placement import choose_candidate, transform_donor_mask
from .quality import aggregate_case_qc, qc_patch
from .runtime import load_inference_runtime
from .sampling import sample_composite_patch, sample_histogram

logger = get_logger(__name__)


def _attempt_seed(base_seed: int, case_index: int, lesion_index: int, attempt: int) -> int:
    return base_seed + case_index * 100_003 + lesion_index * 1_009 + attempt


def _select_donor(entries, target_case: str, rng):
    candidates = [entry for entry in entries if entry["case_id"] != target_case]
    candidates = candidates or entries
    return candidates[int(rng.integers(0, len(candidates)))]


def synthesize(config: dict[str, Any]) -> dict[str, Any]:
    """Generate one cumulative synthetic volume per target case."""
    np = require_numpy()
    synthesis_cfg = config["synthesis"]
    data_cfg = config["data"]
    prepared_dir = Path(data_cfg["prepared_dir"])
    output_dir = Path(synthesis_cfg["output_dir"])
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    masks_dir = output_dir / "masks"
    metadata_dir = output_dir / "metadata"
    for directory in (images_dir, labels_dir, masks_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    requested_per_case = int(synthesis_cfg["num_per_case"])
    max_generation_attempts = int(
        synthesis_cfg.get("max_generation_attempts_per_lesion", 10)
    )
    if requested_per_case <= 0:
        raise ValueError("synthesis.num_per_case must be positive")
    if max_generation_attempts <= 0:
        raise ValueError("synthesis.max_generation_attempts_per_lesion must be positive")

    manifest = read_json(prepared_dir / "manifest.json")
    split = read_json(prepared_dir / "split.json")
    prior = read_json(prepared_dir / "position_prior.json")
    train_entries = [
        entry for entry in manifest["entries"] if entry["split"] == "train"
    ]
    if not train_entries:
        raise RuntimeError("training manifest contains no donor lesions")
    target_split = str(synthesis_cfg.get("split", "train"))
    if target_split not in split["cases"]:
        raise ValueError(f"unknown synthesis split: {target_split}")
    targets = list(split["cases"][target_split])
    model, checkpoint, device, histogram_library = load_inference_runtime(config)
    label_id = int(data_cfg["label_id"])
    base_seed = int(config["seed"])
    records: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []
    rejected_attempts = 0

    for case_index, target_case in enumerate(
        tqdm(targets, desc="Synthesizing", unit="case")
    ):
        source_image_path = image_path(
            data_cfg["source_dataset"], target_case, int(data_cfg["channel"])
        )
        source_label_path = label_path(data_cfg["source_dataset"], target_case)
        original_image, _, source_image = load_ras_with_source(source_image_path)
        original_label, _, source_label = load_ras_with_source(
            source_label_path, label=True
        )
        if original_image.shape != original_label.shape:
            raise ValueError(f"{target_case}: image and label shapes differ")
        current_image = original_image.copy()
        current_label = original_label.copy()
        inserted_mask = np.zeros(current_label.shape, dtype=np.uint8)
        lesion_records: list[dict[str, Any]] = []
        case_rejections = 0

        for lesion_index in range(requested_per_case):
            accepted = False
            for attempt_index in range(max_generation_attempts):
                seed = _attempt_seed(
                    base_seed, case_index, lesion_index, attempt_index
                )
                rng = np.random.default_rng(seed)
                donor = _select_donor(train_entries, target_case, rng)
                try:
                    with np.load(prepared_dir / donor["patch"]) as donor_sample:
                        donor_mask = donor_sample["mask"][0].astype(bool)
                    donor_mask = transform_donor_mask(
                        donor_mask,
                        rng,
                        float(synthesis_cfg["mask_rotation_deg"]),
                        synthesis_cfg["mask_scale_range"],
                    )
                    normalized, normalization = robust_normalize(
                        current_image,
                        float(config["normalization"]["clip_z"]),
                        float(config["normalization"]["foreground_epsilon"]),
                    )
                    center, roi = choose_candidate(
                        current_image,
                        current_label,
                        donor_mask,
                        prior["centers"],
                        rng,
                        protected_dilation=int(synthesis_cfg["protected_dilation"]),
                        max_attempts=int(synthesis_cfg["max_placement_attempts"]),
                    )
                    histogram = sample_histogram(
                        histogram_library,
                        rng,
                        float(synthesis_cfg["histogram_jitter"]),
                    )
                    background = normalized[roi].astype(np.float32)
                    generated, composite = sample_composite_patch(
                        model,
                        background,
                        donor_mask,
                        histogram,
                        device=device,
                        seed=seed,
                        brightness_margin=float(
                            synthesis_cfg.get("brightness_margin", 0.1)
                        ),
                        brightness_transition_voxels=float(
                            synthesis_cfg.get("brightness_transition_voxels", 3.0)
                        ),
                    )
                    qc = qc_patch(
                        background, generated, composite, donor_mask, synthesis_cfg
                    )
                except RuntimeError as exc:
                    case_rejections += 1
                    rejected_attempts += 1
                    logger.debug(
                        "%s lesion %d attempt %d failed: %s",
                        target_case,
                        lesion_index,
                        attempt_index + 1,
                        exc,
                    )
                    continue
                if not qc["passed"]:
                    case_rejections += 1
                    rejected_attempts += 1
                    continue

                composite_raw = denormalize_image(composite, normalization)
                current_roi = current_image[roi].copy()
                current_roi[donor_mask] = composite_raw[donor_mask]
                current_image[roi] = current_roi
                lesion_mask = np.zeros(current_label.shape, dtype=bool)
                lesion_mask[roi] = donor_mask
                inserted_mask[lesion_mask] = 1
                current_label[lesion_mask] = label_id
                lesion_records.append(
                    {
                        "lesion_index": lesion_index,
                        "seed": seed,
                        "attempts": attempt_index + 1,
                        "source_case": donor["case_id"],
                        "source_component": int(donor["component_id"]),
                        "center_ras_voxel": list(center),
                        "histogram": histogram.tolist(),
                        "normalization": normalization,
                        "qc": qc,
                    }
                )
                accepted = True
                break
            if not accepted:
                logger.warning(
                    "%s lesion %d was not generated after %d attempts",
                    target_case,
                    lesion_index,
                    max_generation_attempts,
                )

        if not lesion_records:
            failed_cases.append(
                {
                    "case_id": target_case,
                    "reason": "no_accepted_lesions",
                    "rejected_attempts": case_rejections,
                }
            )
            continue

        sample_id = f"{target_case}_syn"
        image_output = images_dir / f"{sample_id}_0000.nii.gz"
        label_output = labels_dir / f"{sample_id}.nii.gz"
        mask_output = masks_dir / f"{sample_id}.nii.gz"
        synthetic_native = restore_ras_to_source(current_image, source_image)
        label_native = restore_ras_to_source(current_label, source_label)
        mask_native = restore_ras_to_source(inserted_mask, source_label)
        save_like(synthetic_native, source_image, image_output, dtype=np.float32)
        save_like(label_native, source_label, label_output, dtype=np.uint8)
        save_like(mask_native, source_label, mask_output, dtype=np.uint8)

        background_exact = bool(
            np.array_equal(current_image[inserted_mask == 0], original_image[inserted_mask == 0])
        )
        metadata = {
            "sample_id": sample_id,
            "target_case": target_case,
            "requested_lesions": requested_per_case,
            "accepted_lesions": len(lesion_records),
            "complete": len(lesion_records) == requested_per_case,
            "rejected_attempts": case_rejections,
            "lesions": lesion_records,
            "checkpoint": str(synthesis_cfg["checkpoint"]),
            "checkpoint_step": int(checkpoint.get("global_step", -1)),
            "manifest_hash": manifest["hash"],
            "split_hash": split["hash"],
            "qc": aggregate_case_qc(lesion_records, background_exact),
            "outputs": {
                "image": str(image_output),
                "label": str(label_output),
                "inserted_mask": str(mask_output),
            },
        }
        metadata["hash"] = stable_hash(metadata)
        write_json(metadata_dir / f"{sample_id}.json", metadata)
        records.append(metadata)

    complete_cases = sum(bool(record["complete"]) for record in records)
    accepted_lesions = sum(int(record["accepted_lesions"]) for record in records)
    summary = {
        "requested_cases": len(targets),
        "accepted_cases": len(records),
        "failed_cases": failed_cases,
        "complete_cases": complete_cases,
        "partial_cases": len(records) - complete_cases,
        "requested_lesions": len(targets) * requested_per_case,
        "accepted_lesions": accepted_lesions,
        "rejected_attempts": rejected_attempts,
        "completion_rate": complete_cases / max(len(targets), 1),
        "qc_rate": accepted_lesions
        / max(accepted_lesions + rejected_attempts, 1),
        "records": [record["sample_id"] for record in records],
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    logger.info(
        "Synthesis complete: cases=%d/%d, lesions=%d/%d",
        len(records),
        len(targets),
        accepted_lesions,
        len(targets) * requested_per_case,
    )
    return summary
