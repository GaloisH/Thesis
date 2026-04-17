import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

# 导入项目中定义的模块
from model import nnUNet
from preprocess import MeningiomaDataset

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化网络，通道数为 1（例如 T1ce 模态），输出通道数为 1（二分类：背景和病灶）
    model = nnUNet(in_channels=1, out_channels=1, bilinear=False).to(device)
    
    # 损失函数与优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 数据集路径设置
    data_dir = r"D:\python_code\projects\thesis\datasets\raw\BraTS2020_TrainingData"
    os.makedirs(data_dir, exist_ok=True)

    # 从 preprocess.py 中导入的 MeningiomaDataset 自动加载 3D 图像和掩模
    dataset = MeningiomaDataset(data_dir=data_dir)
    
    # 因为可能没有真实数据存在，当数据集为空时使用 Dummy 数据进行测试保证脚本连通性
    if len(dataset) == 0:
        print("未找到真实的 NIfTI 数据，正在使用随机生成的 Dummy 数据来测试训练循环...")
        # 伪造 4 个 batch，每个 batch 大小为 2，维度为 (B, C, H, W)
        dummy_images = torch.randn(2, 1, 240, 240)
        dummy_masks = torch.randint(0, 2, (2, 1, 240, 240)).float()
        dataloader = [(dummy_images, dummy_masks)] * 4
    else:
        # 使用真实的 DataLoader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print("开始训练...")
    epochs = 2
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            if len(dataset) > 0:
                # 预处理返回的 NIfTI 格式是 3D：(B, C, H, W, D)
                # 而 model.py 中的 nnUNet 使用的是 Conv2d，需要 2D 图像：(B, C, H, W)
                # 这里我们提取 3D 体数据中间的 2D 切片来进行 2D 训练
                d_dim = images.shape[-1]
                slice_idx = d_dim // 2
                images = images[..., slice_idx] # 取出中间切片 -> (B, 1, H, W)
                masks = masks[..., slice_idx]
                
                images = images.float()
                masks = masks.float()

            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{4 if len(dataset)==0 else len(dataloader)}], Loss: {loss.item():.4f}")
            
        avg_loss = epoch_loss / (4 if len(dataset)==0 else len(dataloader))
        print(f"Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f}")
        
    print("训练测试完成！")
    
    # 保存模型
    torch.save(model.state_dict(), "nnunet_test_model.pth")
    print("模型权重已保存至 nnunet_test_model.pth")

if __name__ == "__main__":
    train()