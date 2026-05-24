import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

unet_result=r"C:\Users\13765\Downloads\case_000_seg_unet.nii.gz"
swinunetr_result=r"C:\Users\13765\Downloads\case_000_seg.nii.gz"

def visualize(result):
    img=nib.load(result)
    shape=img.shape
    img_data=img.get_fdata()
    img_x=shape[0]//2
    img_y=shape[1]//2
    img_z=shape[2]//2
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_data[img_x, :, :], cmap='gray')
    plt.title('Sagittal View')
    plt.subplot(1, 3, 2)
    plt.imshow(img_data[:, img_y, :], cmap='gray')
    plt.title('Coronal View')
    plt.subplot(1, 3, 3)
    plt.imshow(img_data[:, :, img_z], cmap='gray')
    plt.title('Axial View')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize(unet_result)
    visualize(swinunetr_result)
