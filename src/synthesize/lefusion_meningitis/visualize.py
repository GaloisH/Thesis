import SimpleITK as sitk
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
from logger import get_logger
from tqdm import tqdm

logger = get_logger(__name__)
CONFIG_PATH = r"config/lefusion_meningitis.yaml"


def read_imgs_labels(dir_path: str):
    """
    读取指定目录下的图像、标签和掩码文件，并返回它们的路径列表。
    """
    imgs_dir = os.path.join(dir_path, "images")
    labels_dir = os.path.join(dir_path, "labels")
    masks_dir = os.path.join(dir_path, "masks")
    imgs_list = sorted([os.path.join(imgs_dir, name) for name in os.listdir(imgs_dir)])
    labels_list = sorted(
        [os.path.join(labels_dir, name) for name in os.listdir(labels_dir)]
    )
    masks_list = sorted(
        [os.path.join(masks_dir, name) for name in os.listdir(masks_dir)]
    )
    return imgs_list, labels_list, masks_list


def select_slice(mask: sitk.Image, axis: int = 0):
    """
    从图像、标签和掩码中选择指定切片。
    """
    axes = [(1, 2), (0, 2), (0, 1)]
    mask_array = sitk.GetArrayFromImage(mask)
    mask_sum = mask_array.sum(axis=axes[axis])
    slice_idx = np.argmax(mask_sum)
    return slice_idx


def visualize(ori_img_path, img_path: str, label_path: str, mask_path: str):
    """
    可视化图像、标签和掩码的指定切片。
    """
    img = sitk.ReadImage(img_path)
    label = sitk.ReadImage(label_path)
    mask = sitk.ReadImage(mask_path)

    slice_idx = 0

    ori_img = sitk.ReadImage(ori_img_path)
    ori_img_array = sitk.GetArrayFromImage(ori_img)
    img_array = sitk.GetArrayFromImage(img)
    label_array = sitk.GetArrayFromImage(label)
    mask_array = sitk.GetArrayFromImage(mask)
    plt.figure(figsize=(16, 12))
    num = 1
    for axis in range(3):
        if axis == 0:
            slice_idx = select_slice(mask, axis=0)
            ori_img_slice = ori_img_array[slice_idx, :, :]
            img_slice = img_array[slice_idx, :, :]
            label_slice = label_array[slice_idx, :, :]
            mask_slice = mask_array[slice_idx, :, :]
        elif axis == 1:
            slice_idx = select_slice(mask, axis=1)
            ori_img_slice = ori_img_array[:, slice_idx, :]
            img_slice = img_array[:, slice_idx, :]
            label_slice = label_array[:, slice_idx, :]
            mask_slice = mask_array[:, slice_idx, :]
        else:
            slice_idx = select_slice(mask, axis=2)
            ori_img_slice = ori_img_array[:, :, slice_idx]
            img_slice = img_array[:, :, slice_idx]
            label_slice = label_array[:, :, slice_idx]
            mask_slice = mask_array[:, :, slice_idx]

        plt.subplot(3, 4, num)
        plt.imshow(ori_img_slice, cmap="gray")
        plt.axis("off")
        num += 1
        plt.subplot(3, 4, num)
        plt.imshow(img_slice, cmap="gray")
        plt.contour(mask_slice, levels=[0.5], colors="red", linewidths=0.5, alpha=0.25)
        plt.axis("off")
        num += 1
        plt.subplot(3, 4, num)
        plt.imshow(label_slice, cmap="gray")
        plt.axis("off")
        num += 1

        plt.subplot(3, 4, num)
        plt.imshow(mask_slice, cmap="gray")
        plt.axis("off")
        num += 1

        plt.tight_layout()


def run():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    dir_path = r'datasets\synthesis_0819'
    output_dir = config["visualization"]["output_dir"]
    ori_img_dir_path = config["data"]["source_dataset"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    imgs_list, labels_list, masks_list = read_imgs_labels(dir_path)
    for img_path, label_path, mask_path in tqdm(
        zip(imgs_list, labels_list, masks_list), total=len(imgs_list)
    ):
        file_name = os.path.basename(img_path)
        parts = file_name.split("_")
        parts.pop(2)
        new_name = "_".join(parts)
        ori_img_path = os.path.join(ori_img_dir_path, "imagesTr", new_name)
        visualize(ori_img_path, img_path, label_path, mask_path)
        plt.savefig(
            os.path.join(
                output_dir, os.path.basename(img_path).replace(".nii.gz", ".png")
            ),
            dpi=300,
        )
        plt.close()


if __name__ == "__main__":
    run()
