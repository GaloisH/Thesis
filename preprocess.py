import os
import glob
import numpy as np
import nibabel as nib
import cv2
from torch.utils.data import Dataset, DataLoader

def get_paired_files(data_dir):
    """获取匹配的图像和掩码文件路径"""
    # 使用glob加速文件搜索，并保证路径匹配
    masks = sorted(glob.glob(os.path.join(data_dir, '**', '*_seg.nii'), recursive=True))
    images = [m.replace('_seg.nii', '_t1ce.nii') for m in masks]
    return images, masks

class MeningiomaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.images, self.masks = get_paired_files(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 动态读取NIfTI文件并转为数组，减少内存占用
        image_data = nib.load(self.images[idx]).get_fdata(dtype=np.float32)
        mask_data = nib.load(self.masks[idx]).get_fdata(dtype=np.float32)
        
        img_min, img_max = image_data.min(), image_data.max()
        if img_max > img_min:
            image_data = (image_data - img_min) / (img_max - img_min)

        if self.transform:
            image_data = self.transform(image_data)
            mask_data = self.transform(mask_data)
            
        # 增加通道维度以适配PyTorch (C, H, W, D)
        image_data = np.expand_dims(image_data, axis=0)
        mask_data = np.expand_dims(mask_data, axis=0)

        return image_data, mask_data

def visualize_slice(volume_path, slice_idx=64):
    """可视化指定切片的实用工具"""
    data = nib.load(volume_path).get_fdata(dtype=np.float32)
    slice_data = data[:, :, slice_idx]
    
    slice_data_uint8 = cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    cv2.imshow(f"Slice:{slice_idx}", slice_data_uint8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    dataset=MeningiomaDataset(data_dir=r"D:\python_code\projects\thesis\datasets\raw\BraTS2020_TrainingData")
    print(len(dataset))
    # test_path = r"D:\python_code\projects\thesis\datasets\raw\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_seg.nii"
    # if os.path.exists(test_path):
    #     visualize_slice(test_path, slice_idx=64)