import os
import glob
import json
import nibabel as nib
import numpy as np
from tqdm import tqdm


def collect_case_dirs(data_dir, split_name):
    pattern = os.path.join(data_dir, "**", f"BraTS20_{split_name}_*")
    return sorted(p for p in glob.glob(pattern, recursive=True) if os.path.isdir(p))


def prepare_nnunet_data(data_dir, output_dir, task_id=101, task_name="Meningioma"):
    dataset_name = f"Dataset{task_id:03d}_{task_name}"
    nnunet_raw = os.environ.get("nnUNet_raw", os.path.join(output_dir, "nnUNet_raw"))
    modalities = ["t1", "t1ce", "t2", "flair"]

    # 创建 nnUNetv2 目录结构
    task_dir = os.path.join(nnunet_raw, dataset_name)
    imagesTr = os.path.join(task_dir, "imagesTr")
    labelsTr = os.path.join(task_dir, "labelsTr")
    imagesTs = os.path.join(task_dir, "imagesTs")
    labelsTs = os.path.join(task_dir, "labelsTs")

    for p in [imagesTr, labelsTr, imagesTs, labelsTs]:
        os.makedirs(p, exist_ok=True)

    train_case_dirs = collect_case_dirs(data_dir, "Training")
    val_case_dirs = collect_case_dirs(data_dir, "Validation")
    print(
        f"发现 Training 病例 {len(train_case_dirs)} 个，Validation 病例 {len(val_case_dirs)} 个，开始转换为 nnUNetv2 格式..."
    )

    valid_cases = 0

    for case_dir in tqdm(train_case_dirs, desc="Training"):
        case_name = os.path.basename(case_dir)
        mask_path = os.path.join(case_dir, f"{case_name}_seg.nii")
        image_paths = {
            m: os.path.join(case_dir, f"{case_name}_{m}.nii")
            for m in modalities
        }

        missing_files = [p for p in [mask_path, *image_paths.values()] if not os.path.exists(p)]
        if missing_files:
            print(f"警告：样本缺少模态文件 {missing_files}，已跳过该样本。")
            continue

        case_id = f"case_{valid_cases:03d}"

        for channel_idx, modality in enumerate(modalities):
            out_img = os.path.join(imagesTr, f"{case_id}_{channel_idx:04d}.nii.gz")
            if not os.path.exists(out_img):
                nib.save(nib.load(image_paths[modality]), out_img)

        out_label = os.path.join(labelsTr, f"{case_id}.nii.gz")

        if not os.path.exists(out_label):
            mask_nii = nib.load(mask_path)
            mask_data = mask_nii.get_fdata()
            mask_data[mask_data == 4] = 3
            new_mask = nib.Nifti1Image(
                mask_data.astype(np.uint8), mask_nii.affine, mask_nii.header
            )
            nib.save(new_mask, out_label)

        valid_cases += 1
        if valid_cases % 50 == 0:
            print(f"已处理 {valid_cases} 个有效样本...")

    test_cases = 0
    for case_dir in tqdm(val_case_dirs, desc="Validation"):
        case_name = os.path.basename(case_dir)
        mask_path = os.path.join(case_dir, f"{case_name}_seg.nii")
        image_paths = {
            m: os.path.join(case_dir, f"{case_name}_{m}.nii")
            for m in modalities
        }

        missing_files = [p for p in image_paths.values() if not os.path.exists(p)]
        if missing_files:
            print(f"警告：验证样本缺少模态文件 {missing_files}，已跳过该样本。")
            continue

        case_id = f"test_{test_cases:03d}"
        for channel_idx, modality in enumerate(modalities):
            out_img = os.path.join(imagesTs, f"{case_id}_{channel_idx:04d}.nii.gz")
            if not os.path.exists(out_img):
                nib.save(nib.load(image_paths[modality]), out_img)

        out_label = os.path.join(labelsTs, f"{case_id}.nii.gz")
        if os.path.exists(mask_path) and not os.path.exists(out_label):
            mask_nii = nib.load(mask_path)
            mask_data = mask_nii.get_fdata()
            mask_data[mask_data == 4] = 3
            new_mask = nib.Nifti1Image(
                mask_data.astype(np.uint8), mask_nii.affine, mask_nii.header
            )
            nib.save(new_mask, out_label)
        elif not os.path.exists(mask_path):
            print(f"警告：验证样本 {case_name} 缺少标签文件 {mask_path}，仅保存 imagesTs。")

        test_cases += 1

    # 生成 dataset.json
    dataset_json = {
        "channel_names": {"0": "T1", "1": "T1ce", "2": "T2", "3": "Flair"},
        "labels": {
            "background": 0,
            "necrotic": 1,
            "edema": 2,
            "enhancing_tumor": 3,
        },
        "numTraining": valid_cases,
        "file_ending": ".nii.gz",
    }

    json_path = os.path.join(task_dir, "dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"数据准备完成。请确保设置了正确的环境变量指向 {nnunet_raw}。")
    print(f"dataset.json 已生成到 {json_path}")
    print(f"最终写入训练样本 {valid_cases} 个，测试样本 {test_cases} 个。")


if __name__ == "__main__":
    # 项目根目录相对路径
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "datasets", "raw")
    output_dir = os.path.join(root_dir, "datasets")

    # 把环境变量设置放到系统环境外层
    os.environ["nnUNet_raw"] = os.path.join(output_dir, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(output_dir, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(output_dir, "nnUNet_results")

    prepare_nnunet_data(data_dir, output_dir)
