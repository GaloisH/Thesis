import os
import shutil
import logging
from tqdm import tqdm
import json

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="logs/preprocess.log",
    filemode="a",
)

DATASET_DIR = r"D:\python_code\projects\thesis\datasets\meningitis\nii12345"
MASK_DIR = r"D:\python_code\projects\thesis\datasets\meningitis\12345_1"
OUTPUT_DIR = r"D:\python_code\projects\thesis\datasets"


def build_dataset(dataset_dir, mask_dir, output_dir, task_id="002"):
    '''
    建立nnUNet格式的数据集
    '''
    os.makedirs(os.path.join(output_dir, f"Dataset{task_id}_Meningitis"), exist_ok=True)
    tr_data = os.path.join(
        output_dir, "nnUNet_raw", f"Dataset{task_id}_Meningitis", "imagesTr"
    )
    tr_mask = os.path.join(
        output_dir, "nnUNet_raw", f"Dataset{task_id}_Meningitis", "labelsTr"
    )
    os.makedirs(tr_data, exist_ok=True)
    os.makedirs(tr_mask, exist_ok=True)

    data_list = os.listdir(dataset_dir)
    mask_list = os.listdir(mask_dir)
    name_list = [name.split("_")[0] for name in mask_list]
    nums = len(name_list)
    ids = [i for i in range(nums)]
    name_dict = dict(zip(name_list, ids))
    logging.info(f"Processing {nums} cases for Dataset{task_id}_Meningitis")

    # 处理数据文件
    for data in tqdm(data_list):
        if data.endswith(".nii.gz"):
            data_id = name_dict[data.split("_")[0]]
            modality = (int(data.split("_")[2])) % 10 - 1
            dest_path = os.path.join(
                tr_data,
                f"case_{data_id:03d}_{modality:04d}.nii.gz",
            )
            if not os.path.exists(dest_path):
                shutil.copy(os.path.join(dataset_dir, data), dest_path)

    logging.info(f"Processing {len(mask_list)} masks for Dataset{task_id}_Meningitis")

    # 处理掩码文件
    for mask in tqdm(mask_list):
        if mask.endswith(".nii.gz"):
            mask_id = name_dict[mask.split("_")[0]]
            dest_path = os.path.join(
                tr_mask,
                f"case_{mask_id:03d}.nii.gz",
            )
            if not os.path.exists(dest_path):
                shutil.copy(os.path.join(mask_dir, mask), dest_path)
    logging.info(f"Finished processing Dataset{task_id}_Meningitis")

    with open('config/name_dict.json', 'w') as f:
        json.dump(name_dict, f, indent=4)


if __name__ == "__main__":
    build_dataset(DATASET_DIR, MASK_DIR, OUTPUT_DIR)
