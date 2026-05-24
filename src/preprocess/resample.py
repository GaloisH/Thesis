import os
import SimpleITK as sitk
from tqdm import tqdm

BASE_DIR = r"D:\python_code\projects\thesis\datasets\nnUNet_raw\Dataset002_Meningitis"


def resample_image(image, reference):
    out = sitk.Resample(
        image,
        reference.GetSize(),
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        reference.GetOrigin(),
        reference.GetSpacing(),
        reference.GetDirection(),
    )
    return out


def main(n: int = 30, base_dir: str = BASE_DIR):
    data_dir = os.path.join(BASE_DIR, "imagesTr")
    mask_dir = os.path.join(BASE_DIR, "labelsTr")

    for i in tqdm(range(n)):
        paths=[os.path.join(data_dir, f"case_{i:03d}_000{j}.nii.gz") for j in range(3)]
        imgs = [
            sitk.ReadImage(path)
            for path in paths
        ]
        ref = imgs[0]
        resampled_1=resample_image(imgs[1], ref)
        sitk.WriteImage(resampled_1, paths[1])
        resampled_2=resample_image(imgs[2], ref)
        sitk.WriteImage(resampled_2, paths[2])
        mask_path=os.path.join(mask_dir, f"case_{i:03d}.nii.gz")
        mask_img=sitk.ReadImage(mask_path)
        resampled_mask=resample_image(mask_img, ref)
        sitk.WriteImage(resampled_mask, mask_path)


if __name__ == "__main__":
    main()
