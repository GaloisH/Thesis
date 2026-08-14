"""接受路径端到端验证（计划验证项 5 的缩减版）。

对单个真实病例 case_000 注入与真实影像对齐的合成 FastSurfer 皮层分割，
以临时 prepared 副本（split 仅含该病例）驱动真实 `synthesize()`，
然后核对已接受病灶 metadata：
  - 每条病灶 placement.cortical_fraction == 1.0
  - 病灶不与保护区重叠（由 choose_cortical_candidate 保证，此处复核）
  - 掩膜外背景保持逐体素不变（background_exact）

需要真实数据与模型（prepared/manifest/split/best.pt）。运行方式：
    python verify_cortical_pipeline.py
默认清理临时产物；保留产物可传 --keep。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nibabel.freesurfer.mghformat import MGHImage

from lefusion_meningitis.config import load_config
from lefusion_meningitis.io import load_ras_with_source, read_json
from lefusion_meningitis.synthesis import synthesize

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "lefusion_meningitis.yaml"
TARGET_CASE = "case_000"
CORTICAL_LABEL = 1002


def _build_config(tmp: Path) -> dict:
    """构造指向临时 prepared/快照与临时输出的配置。"""
    overrides = {
        "data": {
            "prepared_dir": str(tmp / "prepared"),
            "fastsurfer_subjects_dir": str(tmp / "fastsurfer_subjects"),
        },
        "synthesis": {
            "output_dir": str(tmp / "out"),
            "num_per_case": 1,
        },
    }
    return load_config(CONFIG_PATH, overrides)


def _prepare_workdir(config: dict, tmp: Path) -> None:
    real_prepared = Path(load_config(CONFIG_PATH)["data"]["prepared_dir"])
    if not (real_prepared / "manifest.json").is_file():
        raise FileNotFoundError(f"prepared dir missing manifest: {real_prepared}")
    # 复制 prepared 并保留 manifest/patches/histograms，仅裁剪 split
    shutil.copytree(real_prepared, tmp / "prepared")
    split = read_json(tmp / "prepared" / "split.json")
    split["cases"]["train"] = [TARGET_CASE]
    split["counts"]["train"] = 1
    (tmp / "prepared" / "split.json").write_text(
        json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # 合成与 case_000 RAS 影像对齐的皮层分割
    source_image_path = Path(config["data"]["source_dataset"]) / "imagesTr" / f"{TARGET_CASE}_{int(config['data']['channel']):04d}.nii.gz"
    _, ras, _ = load_ras_with_source(source_image_path)
    seg = np.zeros(ras.shape, dtype=np.int16)
    lo = np.asarray(ras.shape) // 3
    hi = 2 * np.asarray(ras.shape) // 3
    seg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = CORTICAL_LABEL
    seg_path = (
        tmp / "fastsurfer_subjects" / TARGET_CASE / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
    )
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    MGHImage(seg.astype(np.float32), ras.affine).to_filename(str(seg_path))


def _verify(output_dir: Path) -> None:
    metadata_files = sorted((output_dir / "metadata").glob("*.json"))
    if not metadata_files:
        raise AssertionError("no case metadata produced")
    for meta_path in metadata_files:
        meta = read_json(meta_path)
        assert meta["target_case"] == TARGET_CASE
        assert meta["accepted_lesions"] == 1, meta
        assert meta["complete"], meta
        lesion = meta["lesions"][0]
        placement = lesion["placement"]
        assert placement["cortical_fraction"] == 1.0, placement
        assert placement["cortical_voxels"] == placement["lesion_voxels"], placement
        assert meta["qc"]["background_exact"], meta["qc"]
        assert "fastsurfer_segmentation" in meta and "fastsurfer_lut" in meta
        print(json.dumps(meta, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="keep temp artifacts")
    args = parser.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="cortical_verify_"))
    try:
        config = _build_config(tmp)
        _prepare_workdir(config, tmp)
        summary = synthesize(config)
        print(f"summary: accepted_cases={summary['accepted_cases']} "
              f"accepted_lesions={summary['accepted_lesions']}")
        output_dir = Path(config["synthesis"]["output_dir"])
        _verify(output_dir)
        print("ACCEPT-PATH VERIFICATION PASSED")
        return 0
    finally:
        if args.keep:
            print(f"kept temp artifacts at: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
