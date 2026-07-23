

import os
import shutil
import logging
import json
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="logs/preprocess.log",
    filemode="a",
)

# Source directories containing all data (both labeled and unlabeled)
SOURCE_DIRS = [
    r"D:\python_code\projects\thesis\datasets\meningitis\脑膜MRIniifiles",
    r"D:\python_code\projects\thesis\datasets\meningitis\脑膜补niifiles",
]
NAME_DICT_PATH = r"D:\python_code\projects\thesis\config\name_dict.json"
# Output root
OUTPUT_DIR = r"D:\python_code\projects\thesis\datasets"


def parse_patient_name(filename: str) -> str:
    """
    提取患者姓名
    """
    stem = filename.removesuffix(".nii.gz").removesuffix(".nii")
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse patient name from: {filename}")
    return parts[0]


def parse_case_id(filename: str) -> str:
    """
    提取病例编号
    """
    stem = filename.removesuffix(".nii.gz").removesuffix(".nii")
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse case identifier from: {filename}")
    return f"{parts[0]}_{parts[1]}"


def parse_modality(filename: str) -> int:
    """
    提取模态通道索引

    模态代码是第三个下划线分隔的字段 (0-indexed: parts[2]).
    通道索引 = (模态代码 % 10) - 1

    """
    stem = filename.removesuffix(".nii.gz").removesuffix(".nii")
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse modality code from: {filename}")
    code = int(parts[2])
    return (code % 10) - 1


def load_labeled_patient_names(name_dict_path: str) -> set[str]:
    """
    加载已标注患者姓名集合
    """
    if not os.path.isfile(name_dict_path):
        logging.warning(f"Name dict not found: {name_dict_path}")
        return set()

    with open(name_dict_path, "r", encoding="utf-8") as f:
        name_dict = json.load(f)

    logging.info(f"Loaded {len(name_dict)} labeled patient names from {name_dict_path}")
    return set(name_dict.keys())


def build_test_set(
    source_dirs: list[str],
    name_dict_path: str,
    output_dir: str,
    task_id: str = "003",
):
    """
    建立nnUNet格式的测试集 (imagesTs) 仅包含未标注病例
    """
    # ------------------------------------------------------------------
    # 1.  Load labeled patient names from name_dict.json
    # ------------------------------------------------------------------
    labeled_patients = load_labeled_patient_names(name_dict_path)

    # ------------------------------------------------------------------
    # 2.  Collect all data files from all source directories, grouped by case
    # ------------------------------------------------------------------
    # case_files[case_id] = [(source_path, modality_channel), ...]
    case_files: dict[str, list[tuple[str, int]]] = defaultdict(list)

    for src_dir in source_dirs:
        if not os.path.isdir(src_dir):
            logging.warning(f"Source directory not found, skipping: {src_dir}")
            continue

        for fname in os.listdir(src_dir):
            if not fname.endswith(".nii.gz"):
                continue

            case_id = parse_case_id(fname)
            modality = parse_modality(fname)
            src_path = os.path.join(src_dir, fname)
            case_files[case_id].append((src_path, modality))

    logging.info(
        f"Collected {sum(len(v) for v in case_files.values())} files "
        f"across {len(case_files)} cases from {len(source_dirs)} source directories."
    )

    # ------------------------------------------------------------------
    # 3.  Determine unlabeled cases (patient name NOT in name_dict.json)
    # ------------------------------------------------------------------
    all_cases = sorted(case_files.keys())
    unlabeled_cases = [
        c for c in all_cases if parse_patient_name(c) not in labeled_patients
    ]
    excluded_cases = [c for c in all_cases if c not in unlabeled_cases]

    logging.info(
        f"Total cases: {len(all_cases)} | "
        f"Labeled patients excluded: {len(excluded_cases)} cases | "
        f"Unlabeled (test): {len(unlabeled_cases)} cases"
    )

    if len(unlabeled_cases) == 0:
        logging.warning("No unlabeled cases found — nothing to do.")
        return

    # ------------------------------------------------------------------
    # 4.  Create output directory
    # ------------------------------------------------------------------
    images_ts = os.path.join(
        output_dir, "nnUNet_raw", f"Dataset{task_id}_Meningitis", "imagesTs"
    )
    os.makedirs(images_ts, exist_ok=True)

    # ------------------------------------------------------------------
    # 5.  Build name mapping  {original_case_id → nnunet_identifier}
    # ------------------------------------------------------------------
    name_dict: dict[str, str] = {}
    for idx, case_id in enumerate(unlabeled_cases):
        nnunet_id = f"case_{idx:03d}"
        name_dict[case_id] = nnunet_id

    # Save name mapping for reproducibility
    name_dict_path_out = os.path.join(
        output_dir, "nnUNet_raw", f"Dataset{task_id}_Meningitis", "name_mapping.json"
    )
    with open(name_dict_path_out, "w", encoding="utf-8") as f:
        json.dump(name_dict, f, indent=4, ensure_ascii=False)
    logging.info(f"Name mapping saved to: {name_dict_path_out}")

    # Also save the reverse mapping (nnunet_id → original case_id) for convenience
    reverse_dict = {v: k for k, v in name_dict.items()}
    reverse_dict_path = os.path.join(
        output_dir, "nnUNet_raw", f"Dataset{task_id}_Meningitis", "name_mapping_reverse.json"
    )
    with open(reverse_dict_path, "w", encoding="utf-8") as f:
        json.dump(reverse_dict, f, indent=4, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 6.  Copy files to nnUNet imagesTs format
    # ------------------------------------------------------------------
    total_files = sum(len(case_files[c]) for c in unlabeled_cases)
    logging.info(
        f"Copying {total_files} files for {len(unlabeled_cases)} test cases to imagesTs/"
    )

    skipped = 0
    copied = 0
    for case_id in tqdm(unlabeled_cases, desc="Building test set"):
        nnunet_id = name_dict[case_id]

        # Deduplicate: keep the first file for each modality channel
        # (if the same case appears in multiple source directories)
        seen_modalities: set[int] = set()

        for src_path, modality in case_files[case_id]:
            if modality in seen_modalities:
                logging.debug(f"Skipping duplicate modality {modality} for {case_id}")
                skipped += 1
                continue
            seen_modalities.add(modality)

            dest_name = f"{nnunet_id}_{modality:04d}.nii.gz"
            dest_path = os.path.join(images_ts, dest_name)

            if not os.path.exists(dest_path):
                shutil.copy(src_path, dest_path)
                copied += 1
            else:
                logging.warning(f"Destination already exists, skipping: {dest_path}")
                skipped += 1

    logging.info(
        f"Finished.  Copied: {copied} files | Skipped: {skipped} | "
        f"Test cases: {len(unlabeled_cases)}"
    )

    # ------------------------------------------------------------------
    # 7.  Print summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Test set created successfully!")
    print(f"  Task:          Dataset{task_id}_Meningitis")
    print(f"  Test cases:    {len(unlabeled_cases)}")
    print(f"  Files copied:  {copied}")
    print(f"  Output dir:    {images_ts}")
    print(f"  Name mapping:  {name_dict_path_out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    build_test_set(SOURCE_DIRS, NAME_DICT_PATH, OUTPUT_DIR)
