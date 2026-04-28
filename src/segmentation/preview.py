import numpy as np
import nibabel as nib
import cv2

np.set_printoptions(threshold=np.inf)


def visualize_slice(volume_path, slice_idx=64):
    """可视化指定切片的实用工具"""
    data = nib.load(volume_path).get_fdata(dtype=np.float32)
    slice_data = data[:, :, slice_idx]

    slice_data_uint8 = cv2.normalize(
        slice_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    print(slice_data)
    cv2.imshow(f"Slice:{slice_idx}", slice_data_uint8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    path = r"D:\python_code\projects\thesis\datasets\raw\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_seg.nii"
    visualize_slice(path)

