import nibabel as nib

from nilearn.masking import compute_brain_mask
from nilearn.image import math_img


# 读取T1
t1_img = nib.load(r"D:\python_code\projects\thesis\outputs\lefusion_meningitis\synthetic\images\case_000_syn003_0000.nii.gz")


# 生成brain mask
brain_mask = compute_brain_mask(
    t1_img,
    mask_type="whole-brain"
)


# 保存mask
nib.save(
    brain_mask,
    "brain_mask.nii.gz"
)


# 应用mask
brain_img = math_img(
    "img1 * img2",
    img1=t1_img,
    img2=brain_mask
)


# 保存brain-only MRI
nib.save(
    brain_img,
    "T1_brain.nii.gz"
)