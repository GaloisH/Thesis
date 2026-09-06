"""用 Dataset002 的 fold 0 验证数据覆盖 Dataset101 中的同名病例。"""

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "datasets"
SOURCE = ROOT / "nnUNet_raw" / "Dataset002_Meningitis"
TARGET = ROOT / "nnUNet_raw" / "Dataset101_MeningitisSyn"
SPLITS = ROOT / "nnUNet_preprocessed" / "Dataset002_Meningitis" / "splits_final.json"


with SPLITS.open(encoding="utf-8") as file:
    validation_cases = json.load(file)[0]["val"]

copied_cases = 0
for case_id in validation_cases:
    source_label = SOURCE / "labelsTr" / f"{case_id}.nii.gz"
    target_label = TARGET / "labelsTr" / source_label.name

    # 只覆盖 Dataset101 中已经存在的同名病例。
    if not target_label.exists():
        continue

    for source_image in (SOURCE / "imagesTr").glob(f"{case_id}_*.nii.gz"):
        target_image = TARGET / "imagesTr" / source_image.name
        if target_image.exists():
            shutil.copy2(source_image, target_image)

    shutil.copy2(source_label, target_label)
    copied_cases += 1

print(f"已覆盖 {copied_cases} 个同名病例。")
