import glob
import os

import monai
import torch
import wandb
from monai.data import Dataset, DataLoader
from monai.data.utils import pad_list_data_collate
from monai.networks.nets import SwinUNETR

from plan2transform import build_transforms_from_plan


DEFAULT_CONFIG = {
    "epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "feature_size": 24,
    "drop_rate": 0.2,
    "attn_drop_rate": 0.0,
    "dropout_path_rate": 0.2,
    "use_checkpoint": True,
}


def build_data_dicts(path: str) -> list[dict]:
    """
    将4个模态的路径列表和 mask 路径列表组装成
    [{"image": [t1, t1ce, t2, flair], "label": mask}, ...] 格式。
    LoadImaged 会将 image 列表中的文件沿 channel 维度拼接为 (4, D, H, W)。
    """
    images_t1    = sorted(glob.glob(os.path.join(path, "imagesTr", "*_0000.nii.gz")))
    images_t1ce  = sorted(glob.glob(os.path.join(path, "imagesTr", "*_0001.nii.gz")))
    images_t2    = sorted(glob.glob(os.path.join(path, "imagesTr", "*_0002.nii.gz")))
    images_flair = sorted(glob.glob(os.path.join(path, "imagesTr", "*_0003.nii.gz")))
    images_val_t1  = sorted(glob.glob(os.path.join(path, "imagesTs", "*_0000.nii.gz")))  
    images_val_t1ce  = sorted(glob.glob(os.path.join(path, "imagesTs", "*_0001.nii.gz")))
    images_val_t2  = sorted(glob.glob(os.path.join(path, "imagesTs", "*_0002.nii.gz")))
    images_val_flair  = sorted(glob.glob(os.path.join(path, "imagesTs", "*_0003.nii.gz")))
    masks        = sorted(glob.glob(os.path.join(path, "labelsTr", "*.nii.gz")))

    assert len(images_t1) == len(masks), (
        f"模态文件数 ({len(images_t1)}) 与 mask 数 ({len(masks)}) 不匹配"
    )

    train_data_dicts = [
        {
            "image": [t1, t1ce, t2, flair],
            "label": mask,
        }
        for t1, t1ce, t2, flair, mask in zip(
            images_t1, images_t1ce, images_t2, images_flair, masks
        )
    ]
    val_data_dicts = [
        {
            "image": [t1, t1ce, t2, flair],
            "label": None,
        }
        for t1, t1ce, t2, flair in zip(
            images_val_t1, images_val_t1ce, images_val_t2, images_val_flair
        )
    ]


    return train_data_dicts


def train(path: str, plan_path: str):
    # ------------------------------------------------------------------
    # wandb 初始化（放在 train 内，避免 DataLoader worker 重复创建 run）
    # ------------------------------------------------------------------
    wandb.init(
        project="swinunetr",
        name="swinunetr_100epochs",
        config=DEFAULT_CONFIG,
    )
    config = wandb.config

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------
    train_data_dicts, val_data_dicts = build_data_dicts(path)  

    train_transform, val_transform = build_transforms_from_plan(plan_path)

    # RandCropByPosNegLabeld 每张图返回 num_samples 个 patch（list），
    # pad_list_data_collate 负责将它们正确拼接为一个 batch
    train_ds = Dataset(data=train_data_dicts, transform=train_transform)
    val_ds   = Dataset(data=val_data_dicts,   transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_list_data_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_list_data_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ------------------------------------------------------------------
    # 模型：4 输入通道，3 输出通道（WT / TC / ET 各一个二值图）
    # ------------------------------------------------------------------
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=config.feature_size,
        use_checkpoint=config.use_checkpoint,
        spatial_dims=3,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(7, 7, 7),
        norm_name="instance",
        drop_rate=config.drop_rate,
        attn_drop_rate=config.attn_drop_rate,
        dropout_path_rate=config.dropout_path_rate,
    ).to(device)

    # ------------------------------------------------------------------
    # 损失函数：
    #   label 已经是 (B, 3, D, H, W) 多通道二值图（来自 ConvertToBratsRegionsd）
    #   → 多标签问题，各通道独立 → sigmoid + BCELoss 或 DiceLoss(sigmoid=True)
    #   不能用 softmax（各通道不互斥）也不能用 to_onehot_y（label 已是多通道）
    # ------------------------------------------------------------------
    loss_fn = monai.losses.DiceLoss(
        sigmoid=True,
        to_onehot_y=False,   # label 已是 one-hot 形式，无需再转换
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    epochs = config.epochs

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        epoch_loss = 0.0
        for step, batch_data in enumerate(train_loader, start=1):
            inputs = batch_data["image"].to(device)   # (B, 4, D, H, W)
            labels = batch_data["label"].to(device)   # (B, 3, D, H, W)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs)               # (B, 3, D, H, W)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(
                f"Epoch {epoch + 1}/{epochs}  "
                f"Step {step}/{epoch_len}  "
                f"Loss: {loss.item():.4f}"
            )

        avg_train_loss = epoch_loss / step

        # ---- val ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_step, val_batch in enumerate(val_loader, start=1):
                val_inputs = val_batch["image"].to(device)
                val_labels = val_batch["label"].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    val_outputs = model(val_inputs)
                    val_loss += loss_fn(val_outputs, val_labels).item()
        avg_val_loss = val_loss / val_step

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss":   avg_val_loss,
            "epoch":      epoch + 1,
        })

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  → 保存最优模型（val_loss={best_val_loss:.4f}）")

    wandb.finish()


if __name__ == "__main__":
    plan_path = r"D:\python_code\projects\thesis\datasets\nnUNet_preprocessed\Dataset101_Meningioma\nnUNetPlans.json"
    path      = r"D:\python_code\projects\thesis\datasets\nnUNet_raw\Dataset101_Meningioma"
    train(path, plan_path)
