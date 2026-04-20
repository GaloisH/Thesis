import os
import glob
import shutil
import json
import nibabel as nib
import numpy as np


def prepare_nnunet_data(data_dir, output_dir, task_id=101, task_name="Meningioma"):
    dataset_name = f"Dataset{task_id:03d}_{task_name}"
    nnunet_raw = os.environ.get("nnUNet_raw", os.path.join(output_dir, "nnUNet_raw"))

    # 创建 nnUNetv2 目录结构
    task_dir = os.path.join(nnunet_raw, dataset_name)
    imagesTr = os.path.join(task_dir, "imagesTr")
    labelsTr = os.path.join(task_dir, "labelsTr")
    imagesTs = os.path.join(task_dir, "imagesTs")

    for p in [imagesTr, labelsTr, imagesTs]:
        os.makedirs(p, exist_ok=True)

    # 找到所有的标签文件，并应用对应的图像文件命名规则
    masks = sorted(glob.glob(os.path.join(data_dir, "**", "*_seg.nii"), recursive=True))
    images = [m.replace("_seg.nii", "_t1ce.nii") for m in masks]

    print(f"找到 {len(masks)} 个带标签的样本，开始转换为 nnUNetv2 格式...")

    for idx, (img_path, mask_path) in enumerate(zip(images, masks)):
        if not os.path.exists(img_path):
            print(f"警告：找不到对应的图像文件 {img_path}，已跳过该样本。")
            continue

        case_id = f"case_{idx:03d}"

        out_img = os.path.join(imagesTr, f"{case_id}_0000.nii.gz")
        out_label = os.path.join(labelsTr, f"{case_id}.nii.gz")

        # 将原始 .nii 转换为 .nii.gz
        if not os.path.exists(out_img):
            img_nii = nib.load(img_path)
            nib.save(img_nii, out_img)

        if not os.path.exists(out_label):
            mask_nii = nib.load(mask_path)
            # nnUNet 标签需要是整数格式，二分类标签
            mask_data = mask_nii.get_fdata()
            mask_data[mask_data == 4] = 3
            # 使用 float32 或者 int16 再次包装为 nifti1image 兼容nnunetv2
            new_mask = nib.Nifti1Image(
                mask_data.astype(np.uint8), mask_nii.affine, mask_nii.header
            )
            nib.save(new_mask, out_label)

        if (idx + 1) % 50 == 0:
            print(f"已处理 {idx + 1} 个文件...")

    # 生成 dataset.json
    dataset_json = {
        "channel_names": {"0": "T1ce"},
        "labels": {
            "background": 0,
            "whole_tumor": [1, 2, 3],
            "tumor_core": [2, 3],
            "enhancing_tumor": 3,
        },
        "regions_class_order": [1, 2, 3],
        "numTraining": len(masks),
        "file_ending": ".nii.gz",
    }

    json_path = os.path.join(task_dir, "dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"数据准备完成。请确保设置了正确的环境变量指向 {nnunet_raw}。")
    print(f"dataset.json 已生成到 {json_path}")


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
