from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import os
import cv2
import numpy as np

def get_files(path):
    images=[]
    masks=[]
    for dirpath in os.listdir(path):
        for file in os.listdir(os.path.join(path,dirpath)):
            if file.endswith('_t1ce.nii'):
                image_path=os.path.join(path,dirpath,file)
                images.append(image_path)
            elif file.endswith('_seg.nii'):
                mask_path=os.path.join(path,dirpath,file)
                masks.append(mask_path)

    return images, masks


def get_image(image_path):
    img=nib.load(image_path)
    data = img.get_fdata(dtype=np.float32)
    return data

def get_slice(n,image_path):
    data = get_image(image_path)[:,:,n]  # 正确的切片语法
    data_uint8 = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
    cv2.imshow(f"Slice:{n}",data_uint8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class MeningitsisDataset(Dataset):
    def __init__(self,images,masks,transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        image=self.images[idx]
        mask=self.masks[idx]
        return image,mask

if __name__=='__main__':
    path=r"D:\python_code\projects\thesis\datasets\raw\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_seg.nii" 
#    get_image(path)
    get_slice(64,path)