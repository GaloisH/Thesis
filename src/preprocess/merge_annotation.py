import os
import shutil
from tqdm import tqdm

GT_DIR = r"datasets\修改南大数据"
IMAGE_DIR = r"datasets\nnUNet_raw\Dataset003_Meningitis\imagesTs"
DATASET_DIR = r"datasets\nnUNet_raw\Dataset002_Meningitis"


def merge_annotation(gt_dir, image_dir, dataset_dir):
    # 分割数据集
    image_dir = os.path.join(dataset_dir, "imagesTr")
    mask_dir = os.path.join(dataset_dir, "labelsTr")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    # 数据集大小
    n = len(os.listdir(mask_dir))
    # 遍历图像文件夹
    for file in os.listdir(gt_dir):
        case = file.split(".")[0]
        mask_name_out = f"case_{n:03d}.nii.gz"
        shutil.copy(os.path.join(gt_dir, file), os.path.join(mask_dir, mask_name_out))
        for i in range(3):
            img_name = case + f"_{i:04d}.nii.gz"
            img_name_out = f"case_{n:03d}_{i:04d}.nii.gz"
            img_path = os.path.join(IMAGE_DIR, img_name)
            if os.path.exists(img_path):
                # 复制mask与图像
                shutil.copy(img_path, os.path.join(image_dir, img_name_out))
        n += 1


if __name__ == "__main__":
    merge_annotation(GT_DIR, IMAGE_DIR, DATASET_DIR)
