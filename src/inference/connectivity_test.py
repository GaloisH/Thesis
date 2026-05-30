from skimage.measure import regionprops, label
import numpy as np
import SimpleITK as sitk

np.set_printoptions(threshold=np.inf)


def get_connected_components(mask: np.ndarray):
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    print(len(props))
    return props


def get_match_dict(predict_props, mask_props, threshold=0.3):
    predict_nums = len(predict_props)
    mask_nums = len(mask_props)

    match_dict = {}
    for i in range(predict_nums):
        predict_prop = predict_props[i]
        predict_coords = predict_prop.coords
        predict_coords= list(map(tuple,predict_coords))
        # print(set(predict_coords))
        max_rate = -1
        max_index=0
        overlap_rates = []
        for j in range(mask_nums):
            # 计算IoU
            mask_prop = mask_props[j]
            mask_coords=mask_prop.coords
            mask_coords= list(map(tuple,mask_coords))
            overlap_area = set.intersection(set(mask_coords), set(predict_coords))
            union_area = set.union(set(mask_coords), set(predict_coords))
            overlap_rate = len(overlap_area) / len(union_area)
            overlap_rates.append(overlap_rate)

            if overlap_rate > max_rate:
                max_rate = overlap_rate
                max_index = j
        # 建立指标对应字典
        if max_rate > threshold:
            match_dict.update({i: max_index})
    return match_dict


def get_results(predict_props, mask_props, match_dict: dict):
    predict_nums = len(predict_props)
    mask_nums = len(mask_props)
    TP = len(match_dict)
    precision = TP / predict_nums
    recall = TP / mask_nums
    return precision, recall
    

if __name__ == "__main__":
    img = sitk.ReadImage(
        r"D:\python_code\projects\thesis\datasets\nnUNet_raw\Dataset002_Meningitis\labelsTr\case_000.nii.gz"
    )
    img2 = sitk.ReadImage(
        r"D:\python_code\projects\thesis\datasets\nnUNet_raw\Dataset002_Meningitis\labelsTr\case_008.nii.gz"
    )
    img = sitk.GetArrayFromImage(img)
    img2 = sitk.GetArrayFromImage(img2)
    # img=np.array([[[0, 0, 0, 0, 0],
    #            [0, 1, 1, 0, 0],
    #            [0, 1, 1, 0, 0],
    #            [0, 0, 0, 0, 1],
    #            [0, 0, 0, 1, 1]]])
    # img2=np.array([[[0, 0, 0, 0, 0],
    #            [0, 1, 1, 0, 0],
    #            [0, 1, 1, 0, 0],
    #            [0, 0, 0, 0, 0],
    #            [0, 0, 0, 0, 0]]])
    predict_components = get_connected_components(img)
    mask_components = get_connected_components(img2)
    match_dict=get_match_dict(predict_components,mask_components)
    print(get_results(predict_components,mask_components,match_dict))

    
