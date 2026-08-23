import json
import os
import shutil

GT_DIR = r"datasets\修改2"
IMAGE_DIR = r"datasets\nnUNet_raw\Dataset003_Meningitis\imagesTs"
DATASET_DIR = r"datasets\nnUNet_raw\Dataset002_Meningitis"


def merge_annotation(gt_dir, image_dir, dataset_dir):
    # 分割数据集
    target_image_dir = os.path.join(dataset_dir, "imagesTr")
    mask_dir = os.path.join(dataset_dir, "labelsTr")
    os.makedirs(target_image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # 复制前确保每个标注都有完整的三个模态，避免产生不完整病例
    annotation_files = sorted(file for file in os.listdir(gt_dir) if file.endswith(".nii.gz"))
    missing_images = []
    for file in annotation_files:
        case = file.removesuffix(".nii.gz")
        for i in range(3):
            img_name = case + f"_{i:04d}.nii.gz"
            if not os.path.exists(os.path.join(image_dir, img_name)):
                missing_images.append(img_name)
    if missing_images:
        raise FileNotFoundError(f"缺少源影像: {', '.join(missing_images)}")

    # 数据集大小
    n = len(os.listdir(mask_dir))
    # 遍历图像文件夹
    for file in annotation_files:
        case = file.removesuffix(".nii.gz")
        mask_name_out = f"case_{n:03d}.nii.gz"
        shutil.copy(os.path.join(gt_dir, file), os.path.join(mask_dir, mask_name_out))
        for i in range(3):
            img_name = case + f"_{i:04d}.nii.gz"
            img_name_out = f"case_{n:03d}_{i:04d}.nii.gz"
            img_path = os.path.join(image_dir, img_name)
            shutil.copy(img_path, os.path.join(target_image_dir, img_name_out))
        n += 1

    # 同步 nnUNet 数据集中的训练病例数
    dataset_json_path = os.path.join(dataset_dir, "dataset.json")
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)
    dataset_info["numTraining"] = n
    with open(dataset_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    merge_annotation(GT_DIR, IMAGE_DIR, DATASET_DIR)
