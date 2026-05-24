import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    img_mat=[]
    for j in tqdm(range(0,30)):
        imgs = [sitk.ReadImage(rf"D:\python_code\projects\thesis\datasets\nnUNet_raw\Dataset002_Meningitis\labelsTr\case_{j:03d}.nii.gz") for i in range(0,3)]
        for img in imgs:
            print("第{}个病例的第模态图像信息：".format(j))
            print(img.GetSize())
            print(img.GetSpacing())
            print(img.GetDirection())
            print(img.GetOrigin())
        # img_mat.append(imgs)
    
    # img_mat=np.array(img_mat)
    # print(img_mat.shape)
    # imgs=[sitk.ReadImage(rf"D:\python_code\projects\thesis\datasets\nnUNet_raw\Dataset002_Meningitis\imagesTr\case_000_000{i}.nii.gz") for i in range(0,3)]
    # img=sitk.ReadImage(rf"D:\python_code\projects\thesis\datasets\nnUNet_raw\Dataset002_Meningitis\LabelsTr\case_000.nii.gz")
    # print(img.GetSize())
    # for img in imgs:
    #     out=sitk.Resample(img,
    #                       imgs[0].GetSize(),
    #                       sitk.Transform(),
    #                       sitk.sitkNearestNeighbor,
    #                       imgs[0].GetOrigin(),
    #                       imgs[0].GetSpacing(),
    #                       imgs[0].GetDirection())
    #     print(out.GetSize())
    #     print(out.GetSpacing())
    #     print(out.GetDirection())
    #     print(out.GetOrigin())

        # print(img.GetSize())
        # print(img.GetSpacing())
        # print(img.GetDirection())
        # print(img.GetOrigin())

    # Nifti1Image = nib.Nifti1Image(imgs[0].dataobj, imgs[0].affine)