from skimage.morphology import (
    binary_closing,
    remove_small_objects,
    remove_small_holes,
    ball,
)
import SimpleITK as sitk
import numpy as np
from skimage.measure import regionprops, label
import os
from tqdm import tqdm


def get_connected_components(mask: np.ndarray):
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    print(len(props))
    return props


def closure(label_array: np.ndarray):
    get_connected_components(label_array)
    label_array = remove_small_objects(
        label_array.astype(bool), min_size=10, connectivity=3
    )
    label_array = binary_closing(label_array, ball(1))
    get_connected_components(label_array)
    return label_array.astype(np.uint8)

def main(label_dir,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)
    for filename in tqdm(os.listdir(label_dir)):
        file_path=os.path.join(label_dir,filename)

        label_itk = sitk.ReadImage(file_path)
        label_ = sitk.GetArrayFromImage(label_itk)
        label_ = closure(label_)
        new_label_itk = sitk.GetImageFromArray(label_)
        new_label_itk.CopyInformation(label_itk)
        sitk.WriteImage(
            new_label_itk,
            os.path.join(output_dir,filename),
        )

if __name__ == "__main__":
    label_dir = r"datasets\nnUNet_raw\Dataset002_Meningitis\labelsTr"
    output_dir = r"datasets\labelsTr_closure"
    main(label_dir,output_dir)
