import os
import sys
import glob
import json
import torch
import numpy as np

# 添加 src 到环境变量，以便引入 plan2transform
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from projects.thesis.src.segmentation.plan2transform import _parse_plan

import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    ToTensord,
    SaveImage,
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

def build_infer_transforms(plan_path):
    """
    根据 nnUNetPlans.json 构建推理用的数据预处理管线
    """
    p = _parse_plan(plan_path)
    keys = ["image"]
    
    transforms = Compose([
        # 加载 4 个模态的图像，并沿着通道维度拼接为 (4, D, H, W)
        LoadImaged(keys=keys, image_only=False, ensure_channel_first=True),
        # 统一空间方向
        Orientationd(keys=keys, axcodes="RAS"),
        # 重采样到指定的体素间距
        Spacingd(
            keys=keys,
            pixdim=p["spacing"],
            mode=(p["interp_image"],),
        ),
        # 强度归一化 (仅针对非零区域)
        NormalizeIntensityd(
            keys=keys,
            nonzero=p["use_mask_for_norm"], 
            channel_wise=True,
        ),
        # 转为 Tensor
        ToTensord(keys=keys),
    ])
    return transforms, p["patch_size"]

def get_infer_data(input_dir):
    """
    获取输入目录中的所有图像数据字典
    """
    images_t1 = sorted(glob.glob(os.path.join(input_dir, "*_0000.nii.gz")))
    images_t1ce = sorted(glob.glob(os.path.join(input_dir, "*_0001.nii.gz")))
    images_t2 = sorted(glob.glob(os.path.join(input_dir, "*_0002.nii.gz")))
    images_flair = sorted(glob.glob(os.path.join(input_dir, "*_0003.nii.gz")))
    
    if len(images_t1) == 0:
        return []
    
    assert len(images_t1) == len(images_t1ce) == len(images_t2) == len(images_flair), "输入目录中各模态图像数量不一致！"
    
    data_dicts = []
    for t1, t1ce, t2, flair in zip(images_t1, images_t1ce, images_t2, images_flair):
        # 提取 case_id, 用于后续文件保存
        case_id = os.path.basename(t1).replace("_0000.nii.gz", "")
        data_dicts.append({
            "image": [t1, t1ce, t2, flair],
            "case_id": case_id
        })
    return data_dicts


def main():
    # 路径配置 (可根据运行情况使用 argparser)
    base_dir = "/root/autodl-tmp/Thesis"
    plan_path = os.path.join(base_dir, "datasets/nnUNet_preprocessed/Dataset101_Meningioma/nnUNetPlans.json")
    input_dir = os.path.join(base_dir, "datasets/nnUNet_raw/Dataset101_Meningioma/imagesTs")
    
    # 输出目录配置
    output_seg_dir = os.path.join(base_dir, "prediction_results/swinunetr/seg")
    output_prob_dir = os.path.join(base_dir, "prediction_results/swinunetr/prob")
    model_path = os.path.join(base_dir, "best_model_softmax.pth")
    
    # 创建输出文件夹
    os.makedirs(output_seg_dir, exist_ok=True)
    os.makedirs(output_prob_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 准备数据和数据加载器
    infer_transforms, roi_size = build_infer_transforms(plan_path)
    data_dicts = get_infer_data(input_dir)
    
    if not data_dicts:
        print(f"Warning: No valid data found in {input_dir}. Please check your path.")
        return
        
    infer_ds = Dataset(data=data_dicts, transform=infer_transforms)
    infer_loader = DataLoader(infer_ds, batch_size=1, shuffle=False, num_workers=2)
    
    # 2. 构建 SwinUNETR 模型 (需保持与训练参数一致)
    model = SwinUNETR(
        in_channels=4,
        out_channels=4,
        feature_size=48,
        use_checkpoint=True,
        spatial_dims=3,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(7, 7, 7),
        norm_name="instance",
        drop_rate=0.2,
        attn_drop_rate=0.0,
        dropout_path_rate=0.2,
    ).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Exiting.")
        return
        
    # 加载模型权重并设置验证模式
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. 定义保存器 SaveImage
    seg_saver = SaveImage(
        output_dir=output_seg_dir, 
        output_postfix="", 
        output_ext=".nii.gz", 
        separate_folder=False, 
        print_log=True
    )
    prob_saver = SaveImage(
        output_dir=output_prob_dir, 
        output_postfix="prob", 
        output_ext=".nii.gz", 
        separate_folder=False, 
        print_log=True
    )
    
    # 4. 执行推理过程
    max_cases = 2  # 设置最大推理case数，例如设为 5 则只推理前5个，None表示全部推理
    with torch.no_grad():
        for i, batch_data in enumerate(infer_loader):
            if max_cases is not None and i >= max_cases:
                break
            
            case_id = batch_data["case_id"][0]
            print(f"Processing Inference: {case_id}")
            
            inputs = batch_data["image"].to(device)
            # 使用 meta 信息保存原样空间特征，以防 monai 的 metadata dict 支持不同而报错
            meta_dict = None
            if "image_meta_dict" in batch_data:
                meta_dict = {"filename_or_obj": f"{case_id}.nii.gz"}

            # 滑动窗口推理
            outputs = sliding_window_inference(
                inputs=inputs,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )
            
            # 使用 softmax 获取 4 个通道类的概率，形状: (1, 4, D, H, W)
            probs = torch.softmax(outputs, dim=1)
            # 通过 argmax 获得硬分割结果（类别 0-3），形状: (1, 1, D, H, W)
            pred_idx = torch.argmax(probs, dim=1, keepdim=True)
            
            # 由于 batch_size=1，提取单样本的 tensor 并移至 cpu 进行保存
            # 保存硬分割文件 (nii.gz)
            seg_saver(pred_idx[0].cpu(), meta_data=meta_dict)
            
            # 保存 softmax 概率图文件 (nii.gz) 
            prob_saver(probs[0].cpu(), meta_data=meta_dict)

if __name__ == "__main__":
    main()
