from skimage.measure import regionprops, label
import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm

np.set_printoptions(threshold=np.inf)


def get_connected_components(mask: np.ndarray):
    """
    获取二值掩码中的连通组件属性
    """
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    return props


def get_match_dict(predict_props, mask_props, threshold=0.3):
    predict_nums = len(predict_props)
    mask_nums = len(mask_props)

    overlap = 0
    match_dict = {}
    for i in range(predict_nums):
        predict_prop = predict_props[i]
        predict_coords = predict_prop.coords
        predict_coords = list(map(tuple, predict_coords))
        max_rate = -1
        max_index = 0
        overlap_rates = []
        for j in range(mask_nums):
            # 计算IoU
            mask_prop = mask_props[j]
            mask_coords = mask_prop.coords
            mask_coords = list(map(tuple, mask_coords))
            overlap_area = set.intersection(set(mask_coords), set(predict_coords))
            union_area = set.union(set(mask_coords), set(predict_coords))
            overlap_rate = len(overlap_area) / len(union_area)
            overlap_rates.append(overlap_rate)

            if overlap_rate > max_rate:
                max_rate = overlap_rate
                max_index = j
        # 建立指标对应字典
        if max_rate > threshold:
            match_dict.update({i: (max_index, max_rate)})
            overlap += max_rate
    return match_dict, overlap / predict_nums


def get_results(predict_props, mask_props, match_dict: dict):
    predict_nums = len(predict_props)
    mask_nums = len(mask_props)
    TP = len(match_dict)
    precision = TP / predict_nums
    recall = TP / mask_nums
    return precision, recall


def cal_dice(predict_array: np.ndarray, mask_array: np.ndarray):
    """
    计算二值掩码之间的Dice相似系数
    """
    intersection = np.sum(predict_array * mask_array)
    return 2 * intersection / (np.sum(predict_array) + np.sum(mask_array))


def cal_avg_score(predict_dir, mask_dir, validation_list, threshold=0.25):
    avg_precision = 0
    avg_recall = 0
    avg_overlap = 0
    avg_dice = 0

    avg_val_precision = 0
    avg_val_recall = 0
    avg_val_overlap = 0
    avg_val_dice = 0

    avg_train_precision = 0
    avg_train_recall = 0
    avg_train_overlap = 0
    avg_train_dice = 0
    total_num = 0
    for predict_file in tqdm(os.listdir(predict_dir)):
        if predict_file.endswith(".nii.gz"):
            total_num += 1
            # 读取预测文件和对应的mask文件
            predict_path = os.path.join(predict_dir, predict_file)
            mask_path = os.path.join(mask_dir, predict_file)
            predict_itk = sitk.ReadImage(predict_path)
            mask_itk = sitk.ReadImage(mask_path)
            predict_array = sitk.GetArrayFromImage(predict_itk)
            mask_array = sitk.GetArrayFromImage(mask_itk)
            # 获取连通组件属性并计算匹配关系和指标
            predict_props = get_connected_components(predict_array)
            mask_props = get_connected_components(mask_array)
            match_dict, overlap = get_match_dict(
                predict_props, mask_props, threshold=threshold
            )
            precision, recall = get_results(predict_props, mask_props, match_dict)
            dice = cal_dice(predict_array, mask_array)
            print(
                f"{predict_file} -> precision: {precision:.4f}, recall: {recall:.4f}, average overlap: {overlap:.4f}, dice: {dice:.4f}"
            )
            avg_precision += precision
            avg_recall += recall
            avg_overlap += overlap
            avg_dice += dice
            if int(predict_file.split("_")[1].split(".")[0]) in validation_list:
                print(f"Validation case {predict_file} -> precision: {precision:.4f}, recall: {recall:.4f}, average overlap: {overlap:.4f}, dice: {dice:.4f}")
                avg_val_precision += precision
                avg_val_recall += recall
                avg_val_overlap += overlap
                avg_val_dice += dice
            else:
                avg_train_precision += precision
                avg_train_recall += recall
                avg_train_overlap += overlap
                avg_train_dice += dice

    print(f"Average precision: {avg_precision / total_num:.4f}")
    print(f"Average recall: {avg_recall / total_num:.4f}")
    print(f"Average overlap: {avg_overlap / total_num:.4f}")
    print(f"Average dice: {avg_dice / total_num:.4f}")

    print(f"Average validation precision: {avg_val_precision / len(validation_list):.4f}")
    print(f"Average validation recall: {avg_val_recall / len(validation_list):.4f}")
    print(f"Average validation overlap: {avg_val_overlap / len(validation_list):.4f}")
    print(f"Average validation dice: {avg_val_dice / len(validation_list):.4f}")

    print(f"Average training precision: {avg_train_precision / (total_num - len(validation_list)):.4f}")
    print(f"Average training recall: {avg_train_recall / (total_num - len(validation_list)):.4f}")
    print(f"Average training overlap: {avg_train_overlap / (total_num - len(validation_list)):.4f}")
    print(f"Average training dice: {avg_train_dice / (total_num - len(validation_list)):.4f}")


if __name__ == "__main__":
    # img = sitk.ReadImage(
    #     r"D:\python_code\projects\thesis\datasets\nnUNet_raw\Dataset002_Meningitis\labelsTr\case_000.nii.gz"
    # )
    # img2 = sitk.ReadImage(
    #     r"D:\python_code\projects\thesis\datasets\nnUNet_raw\Dataset002_Meningitis\labelsTr\case_008.nii.gz"
    # )
    # img = sitk.GetArrayFromImage(img)
    # img2 = sitk.GetArrayFromImage(img2)
    # # img=np.array([[[0, 0, 0, 0, 0],
    # #            [0, 1, 1, 0, 0],
    # #            [0, 1, 1, 0, 0],
    # #            [0, 0, 0, 0, 1],
    # #            [0, 0, 0, 1, 1]]])
    # # img2=np.array([[[0, 0, 0, 0, 0],
    # #            [0, 1, 1, 0, 0],
    # #            [0, 1, 1, 0, 0],
    # #            [0, 0, 0, 0, 0],
    # #            [0, 0, 0, 0, 0]]])
    # predict_components = get_connected_components(img)
    # mask_components = get_connected_components(img2)
    # match_dict, avg_overlap = get_match_dict(predict_components, mask_components)
    # print(get_results(predict_components, mask_components, match_dict))
    predict_dir = r"datasets\nnUNet_raw\2"
    mask_dir = r"datasets\nnUNet_raw\Dataset002_Meningitis\labelsTr"
    cal_avg_score(predict_dir, mask_dir,[0], threshold=0.25)
