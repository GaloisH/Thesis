from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityd,
    ToTensord,
)
import glob
import os

IMAGE_DIR = r"D:\python_code\projects\thesis\src\synthesize\lesion_ldm\data\0721"
LABEL_DIR = r"D:\python_code\projects\thesis\src\synthesize\lesion_ldm\data\0721"


def create_monai_dataset(image_dir, label_dir) -> Dataset:
    images = sorted(glob.glob(os.path.join(image_dir, "*_img.nii.gz")))
    labels = sorted(glob.glob(os.path.join(label_dir, "*_mask.nii.gz")))
    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            ToTensord(keys=["image", "label"]),
        ]
    )

    # 创建Dataset
    train_ds = Dataset(data=data_dicts, transform=train_transforms)
    return train_ds


if __name__ == "__main__":
    create_monai_dataset(IMAGE_DIR, LABEL_DIR)
