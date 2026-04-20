import blosc2 as bl
import numpy as np

np.set_printoptions(threshold=np.inf)


def convert_b2nd_to_nii(b2nd_path, nii_path):
    data = bl.open(b2nd_path)
    # print(data.info)
    print(data[:])


if __name__ == "__main__":
    path = r"D:\python_code\projects\thesis\datasets\nnUNet_preprocessed\Dataset101_Meningioma\nnUNetPlans_2d\case_000.b2nd"
    convert_b2nd_to_nii(path, None)

