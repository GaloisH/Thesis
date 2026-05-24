"""
Core utilities for lesion frequency/probability map building with ANTs registration.
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ants
import nibabel as nib
import numpy as np
import psutil
from nibabel.processing import resample_from_to
from tqdm import tqdm


@dataclass
class Config:
    dataset_dir: str = r"D:\python_code\projects\thesis\datasets\nnUNet_raw\Dataset101_Meningioma"
    output_dir: str = r"D:\python_code\projects\thesis\datasetsregistration"
    modality: str = "T1"
    n_jobs: int = -1
    enable_checkpoint: bool = True
    ants_threads_per_worker: int = 1
    max_workers_cap: int = 8
    mem_per_worker_gb: float = 4.0

    @property
    def images_dir(self) -> str:
        return os.path.join(self.dataset_dir, "imagesTr")

    @property
    def labels_dir(self) -> str:
        return os.path.join(self.dataset_dir, "labelsTr")

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.output_dir, "checkpoint.json")


DEFAULT_CONFIG = Config()


def _set_thread_limit(n: int) -> None:
    n = str(max(1, n))
    for key in ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = n


def _init_worker(n_threads: int) -> None:
    _set_thread_limit(n_threads)


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _save_checkpoint(path: str, processed: set) -> None:
    _write_json(path, {"completed": sorted(processed)})


def choose_n_jobs(requested: int, cfg: Config) -> int:
    cpu = multiprocessing.cpu_count()
    target = cpu if requested == -1 else max(1, requested)
    available_gb = psutil.virtual_memory().available / (1024**3)
    mem_limit = max(1, int(available_gb // cfg.mem_per_worker_gb))
    return max(1, min(target, cpu, mem_limit, cfg.max_workers_cap))


def load_case_ids(labels_dir: str, max_cases: int = 0) -> List[str]:
    ids = sorted(Path(p).name.removesuffix(".nii.gz") for p in Path(labels_dir).glob("*.nii.gz"))
    return ids[:max_cases] if max_cases > 0 else ids


def resolve_mni_template() -> str:
    from templateflow.api import get as tflow_get
    path = tflow_get(template="MNI152NLin2009cAsym", resolution=2, desc="brain",
                     suffix="T1w", extension=".nii.gz")
    if isinstance(path, (list, tuple)):
        path = path[0] if path else None
    if not path or not os.path.exists(str(path)):
        raise FileNotFoundError(f"Template not found: {path}")
    return str(path)


def resolve_channel_index(dataset_json: str, modality: str) -> int:
    info = _read_json(dataset_json)
    channels = info.get("channel_names", {})
    target = modality.lower()
    for idx, name in channels.items():
        if str(name).lower() == target:
            return int(idx)
    raise ValueError(f"Modality {modality} not found in channel_names")


def register_to_mni(moving_path: str, fixed_path: str) -> dict:
    moving = ants.image_read(moving_path)
    fixed = ants.image_read(fixed_path)
    return ants.registration(
        fixed=fixed, moving=moving, type_of_transform="SyN",
        grad_step=0.1, flow_sigma=1.0, total_sigma=0,
        syn_metric="mattes", syn_sampling=32,
        reg_iterations=(100, 70, 50, 20), verbose=False,
    )


def apply_transform(mask_path: str, fixed_path: str, transforms: list) -> ants.ANTsImage:
    return ants.apply_transforms(
        fixed=ants.image_read(fixed_path),
        moving=ants.image_read(mask_path),
        transformlist=transforms,
        interpolator="nearestNeighbor",
    )


def binarize_mask(img: ants.ANTsImage, thresh: float = 0.5) -> ants.ANTsImage:
    return img.new_image_like((img.numpy() > thresh).astype(np.float32))


@dataclass
class PatientResult:
    case_id: str
    mask_arr: Optional[np.ndarray] = None
    success: bool = False


def process_patient(case_id: str, cfg: Config, channel_idx: int, mni: str) -> PatientResult:
    moving = os.path.join(cfg.images_dir, f"{case_id}_{channel_idx:04d}.nii.gz")
    mask = os.path.join(cfg.labels_dir, f"{case_id}.nii.gz")
    result = PatientResult(case_id)

    if not os.path.exists(moving) or not os.path.exists(mask):
        return result

    _set_thread_limit(cfg.ants_threads_per_worker)
    out_dir = os.path.join(cfg.output_dir, "registered")
    os.makedirs(out_dir, exist_ok=True)

    out_mask = os.path.join(out_dir, f"{case_id}_mask_mni.nii.gz")
    out_t1 = os.path.join(out_dir, f"{case_id}_{cfg.modality.lower()}_mni.nii.gz")

    if os.path.exists(out_mask):
        result.mask_arr = (nib.load(out_mask).get_fdata() > 0.5).astype(np.float32)
        result.success = True
        return result

    try:
        reg = register_to_mni(moving, mni)
        ants.image_write(reg["warpedmovout"], out_t1)
        warped = apply_transform(mask, mni, reg["fwdtransforms"])
        binary = binarize_mask(warped)
        ants.image_write(binary, out_mask)
        result.mask_arr = binary.numpy()
        result.success = True
    except Exception as e:
        print(f"  Failed {case_id}: {e}")

    return result


def build_maps(case_ids: List[str], cfg: Config, channel_idx: int, mni: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    n_jobs = choose_n_jobs(cfg.n_jobs, cfg)
    print(f"Parallel: n_jobs={n_jobs}, threads/worker={cfg.ants_threads_per_worker}")

    template = nib.load(mni)
    shape, affine = template.shape[:3], template.affine
    freq_map = np.zeros(shape, dtype=np.float32)
    n_valid = 0
    processed = set()

    def load_cached(cid: str) -> Optional[np.ndarray]:
        path = os.path.join(cfg.output_dir, "registered", f"{cid}_mask_mni.nii.gz")
        if os.path.exists(path):
            return (nib.load(path).get_fdata() > 0.5).astype(np.float32)
        return None

    if cfg.enable_checkpoint and os.path.exists(cfg.checkpoint_path):
        cached_ids = set(_read_json(cfg.checkpoint_path).get("completed", []))
        for cid in case_ids:
            if cid in cached_ids:
                mask = load_cached(cid)
                if mask is not None and mask.shape == shape:
                    freq_map += mask
                    n_valid += 1
                    processed.add(cid)

    remaining = [c for c in case_ids if c not in processed]
    print(f"Cached: {len(processed)}, Remaining: {len(remaining)}")

    def accumulate(res: PatientResult) -> None:
        nonlocal freq_map, n_valid, processed
        if res.success and res.mask_arr is not None and res.mask_arr.shape == shape:
            freq_map += res.mask_arr
            n_valid += 1
            processed.add(res.case_id)

    if remaining and n_jobs == 1:
        for cid in tqdm(remaining, desc="Registration"):
            accumulate(process_patient(cid, cfg, channel_idx, mni))
            if len(processed) % 5 == 0 and cfg.enable_checkpoint:
                _save_checkpoint(cfg.checkpoint_path, processed)

    elif remaining:
        func = partial(process_patient, cfg=cfg, channel_idx=channel_idx, mni=mni)
        pending = list(remaining)
        ctx = multiprocessing.get_context("spawn")

        try:
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx,
                                      initializer=_init_worker, initargs=(cfg.ants_threads_per_worker,)) as ex:
                futures = {ex.submit(func, c): c for c in pending}
                for fut in tqdm(as_completed(futures), total=len(pending), desc="Parallel"):
                    cid = futures[fut]
                    try:
                        accumulate(fut.result(timeout=3600))
                    except Exception as e:
                        print(f"  Failed {cid}: {e}")
                    pending.remove(cid)
                    if len(processed) % 5 == 0 and cfg.enable_checkpoint:
                        _save_checkpoint(cfg.checkpoint_path, processed)
        except BrokenProcessPool:
            print("Pool crashed, fallback to single process")
            for cid in tqdm(pending, desc="Fallback"):
                accumulate(process_patient(cid, cfg, channel_idx, mni))
                if len(processed) % 5 == 0 and cfg.enable_checkpoint:
                    _save_checkpoint(cfg.checkpoint_path, processed)

    if cfg.enable_checkpoint:
        _save_checkpoint(cfg.checkpoint_path, processed)

    prob_map = freq_map / max(n_valid, 1)
    os.makedirs(cfg.output_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(freq_map, affine), os.path.join(cfg.output_dir, "lesion_frequency_map.nii.gz"))
    nib.save(nib.Nifti1Image(prob_map, affine), os.path.join(cfg.output_dir, "lesion_probability_map.nii.gz"))

    print(f"Done: {n_valid}/{len(case_ids)} valid cases")
    return freq_map, nib.Nifti1Image(prob_map, affine)


def resolve_aal_atlas(output_dir: str) -> Tuple[str, str]:
    """Fetch AAL atlas matching MNI152NLin2009cAsym space."""
    os.makedirs(output_dir, exist_ok=True)
    labels_path = os.path.join(output_dir, "aal_label_names.json")

    # TemplateFlow AAL (preferred)
    try:
        from templateflow.api import get as tflow_get
        path = None
        for desc in [None, "AAL"]:
            result = tflow_get(template="MNI152NLin2009cAsym", resolution=2,
                               atlas="AAL", suffix="dseg", extension=".nii.gz",
                               raise_empty=False, **({"desc": desc} if desc else {}))
            if result:
                path = str(result[0] if isinstance(result, (list, tuple)) else result)
                break
        if path and os.path.exists(path):
            print(f"[INFO] TemplateFlow AAL: {path}")
            mapping: Dict[int, str] = {}
            tsv = path.replace(".nii.gz", ".tsv")
            if os.path.exists(tsv):
                with open(tsv, encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    id_col = next((c for c in (reader.fieldnames or []) if c.lower() in {"index", "label", "id"}), None)
                    name_col = next((c for c in (reader.fieldnames or []) if c.lower() in {"name", "region"}), None)
                    for row in reader:
                        if id_col and name_col:
                            mapping[int(row[id_col])] = row[name_col]
            if not mapping:
                arr = nib.load(path).get_fdata()
                for rid in np.unique(np.rint(arr).astype(np.int32)):
                    if rid > 0:
                        mapping[int(rid)] = f"AAL_Region_{int(rid)}"
            _write_json(labels_path, mapping)
            return path, labels_path
    except Exception as e:
        print(f"[WARN] TemplateFlow AAL failed: {e}")

    # nilearn fallback
    print("[WARN] Using nilearn AAL1 (MNI152Lin) - expect ~1-2mm offset")
    from nilearn.datasets import fetch_atlas_aal
    atlas = fetch_atlas_aal()
    path = str(atlas.maps)
    mapping = {int(i): str(n) for i, n in zip(atlas.indices or [], atlas.labels or [])}
    _write_json(labels_path, mapping)
    return path, labels_path


def _find_csv_cols(fieldnames: List[str]) -> Tuple[Optional[str], Optional[str]]:
    names = [c.lower() for c in fieldnames]
    id_col = fieldnames[next((i for i, n in enumerate(names) if n in {"id", "label", "label_id"}), -1)] if any(n in {"id", "label", "label_id"} for n in names) else None
    name_col = fieldnames[next((i for i, n in enumerate(names) if n in {"name", "region", "label_name"}), -1)] if any(n in {"name", "region", "label_name"} for n in names) else None
    return id_col, name_col


def load_labels(path: Optional[str]) -> Dict[int, str]:
    if not path or not os.path.exists(path):
        return {}
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return {int(k): str(v) for k, v in _read_json(path).items()}
    if suffix == ".csv":
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            id_col, name_col = _find_csv_cols(reader.fieldnames or [])
            if not id_col or not name_col:
                raise ValueError("CSV needs id/name columns")
            return {int(row[id_col]): row[name_col] for row in reader}
    raise ValueError("Use .json or .csv for labels")


def compute_region_probs(prob_path: str, atlas_path: str, csv_path: str,
                         labels_path: Optional[str] = None, min_size: int = 10) -> List[Dict]:
    prob_img = nib.load(prob_path)
    atlas_img = nib.load(atlas_path)
    if prob_img.shape[:3] != atlas_img.shape[:3] or not np.allclose(prob_img.affine, atlas_img.affine, atol=1e-4):
        atlas_img = resample_from_to(atlas_img, (prob_img.shape, prob_img.affine), order=0)

    prob = prob_img.get_fdata().astype(np.float32)
    labels = np.rint(atlas_img.get_fdata()).astype(np.int32)
    names = load_labels(labels_path)

    rows = []
    for rid in np.unique(labels):
        if rid <= 0:
            continue
        mask = labels == rid
        voxels = int(np.count_nonzero(mask))
        if voxels < min_size:
            continue
        probs = prob[mask]
        rows.append({
            "region_id": int(rid),
            "region_name": names.get(int(rid), f"Region_{int(rid)}"),
            "voxel_count": voxels,
            "mean_probability": float(np.mean(probs)),
            "max_probability": float(np.max(probs)),
            "nonzero_voxel_ratio": float(np.mean(probs > 0)),
        })

    rows.sort(key=lambda x: x["mean_probability"], reverse=True)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["region_id", "region_name", "voxel_count",
                                               "mean_probability", "max_probability", "nonzero_voxel_ratio"])
        writer.writeheader()
        writer.writerows(rows)
    return rows