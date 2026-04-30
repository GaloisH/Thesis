import glob
import json
import os

import monai
import torch
import wandb
from monai.data import Dataset, DataLoader
from monai.data.utils import pad_list_data_collate
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

from plan2transform import build_transforms_from_plan
import argparse


DEFAULT_CONFIG = {
    "epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "feature_size": 48,
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
    images_t1 = sorted(glob.glob(os.path.join(path, "imagesTr", "*_0000.nii.gz")))
    images_t1ce = sorted(glob.glob(os.path.join(path, "imagesTr", "*_0001.nii.gz")))
    images_t2 = sorted(glob.glob(os.path.join(path, "imagesTr", "*_0002.nii.gz")))
    images_flair = sorted(glob.glob(os.path.join(path, "imagesTr", "*_0003.nii.gz")))
    images_val_t1 = sorted(glob.glob(os.path.join(path, "imagesTs", "*_0000.nii.gz")))
    images_val_t1ce = sorted(glob.glob(os.path.join(path, "imagesTs", "*_0001.nii.gz")))
    images_val_t2 = sorted(glob.glob(os.path.join(path, "imagesTs", "*_0002.nii.gz")))
    images_val_flair = sorted(glob.glob(os.path.join(path, "imagesTs", "*_0003.nii.gz")))
    masks = sorted(glob.glob(os.path.join(path, "labelsTr", "*.nii.gz")))
    val_masks = sorted(glob.glob(os.path.join(path, "labelsTs", "*.nii.gz")))

    assert len(images_t1) == len(images_t1ce) == len(images_t2) == len(images_flair), (
        "训练集四个模态数量不一致"
    )
    assert len(images_t1) == len(masks), f"训练集模态数 ({len(images_t1)}) 与 mask 数 ({len(masks)}) 不匹配"

    train_data_dicts = [
        {
            "image": [t1, t1ce, t2, flair],
            "label": mask,
        }
        for t1, t1ce, t2, flair, mask in zip(
            images_t1, images_t1ce, images_t2, images_flair, masks
        )
    ]
    val_data_dicts = []
    has_val_images = len(images_val_t1) > 0
    if has_val_images:
        assert len(images_val_t1) == len(images_val_t1ce) == len(images_val_t2) == len(images_val_flair), (
            "验证集四个模态数量不一致"
        )

        # 仅当 labelsTs 存在且数量匹配时，才把 imagesTs 当作验证集。
        if len(val_masks) == len(images_val_t1):
            val_data_dicts = [
                {
                    "image": [t1, t1ce, t2, flair],
                    "label": mask,
                }
                for t1, t1ce, t2, flair, mask in zip(
                    images_val_t1, images_val_t1ce, images_val_t2, images_val_flair, val_masks
                )
            ]


    return train_data_dicts, val_data_dicts


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
    if len(val_data_dicts) == 0:
        # imagesTs 没有标签时，使用训练集切分验证集
        split_idx = int(0.85 * len(train_data_dicts))
        val_data_dicts = train_data_dicts[split_idx:]
        train_data_dicts = train_data_dicts[:split_idx]

    if len(val_data_dicts) == 0:
        raise RuntimeError("验证集为空：请提供带标签的验证集，或保证训练集样本数足够进行切分。")

    train_transform, val_transform = build_transforms_from_plan(plan_path)
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    roi_size = tuple(plan["configurations"]["3d_fullres"]["patch_size"])

    # RandCropByPosNegLabeld 每张图返回 num_samples 个 patch（list），
    # pad_list_data_collate 负责将它们正确拼接为一个 batch
    train_ds = Dataset(data=train_data_dicts, transform=train_transform)
    val_ds   = Dataset(data=val_data_dicts,   transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=pad_list_data_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 模型：4 输入通道，4 输出通道（background/necrotic/edema/enhancing）
    # ------------------------------------------------------------------
    model = SwinUNETR(
        in_channels=4,
        out_channels=4,
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
    #   label 已经是 (B, 4, D, H, W) 互斥 one-hot（来自 ConvertToBratsRegionsd）
    #   → 多类别互斥分割，使用 softmax
    # ------------------------------------------------------------------
    loss_fn = monai.losses.DiceCELoss(
        softmax=True,
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
    class_names = ["background", "necrotic", "edema", "enhancing_tumor"]
    eps = 1e-8
    # FIX 2: 提到循环外，避免每个 step 重复计算
    epoch_len = len(train_loader)

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        epoch_loss = 0.0
        for step, batch_data in enumerate(train_loader, start=1):
            inputs = batch_data["image"].to(device)   # (B, 4, D, H, W)
            labels = batch_data["label"].to(device)   # (B, 4, D, H, W)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)               # (B, 4, D, H, W)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(
                f"Epoch {epoch + 1}/{epochs}  "
                f"Step {step}/{epoch_len}  "
                f"Loss: {loss.item():.4f}"
            )

        avg_train_loss = epoch_loss / step

        # ---- val ----
        model.eval()
        val_loss = 0.0
        sum_inter = torch.zeros(4, device=device)
        sum_pred = torch.zeros(4, device=device)
        sum_label = torch.zeros(4, device=device)
        with torch.no_grad():
            for val_step, val_batch in enumerate(val_loader, start=1):
                val_inputs = val_batch["image"].to(device)
                val_labels = val_batch["label"].to(device)
                val_outputs = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=roi_size,
                    sw_batch_size=2,
                    predictor=model,
                    overlap=0.5,
                )
                val_loss += loss_fn(val_outputs, val_labels).item()

                # 互斥多类别：softmax 后 argmax，再还原为 one-hot 便于按类统计 Dice
                probs = torch.softmax(val_outputs, dim=1)
                pred_idx = torch.argmax(probs, dim=1, keepdim=True)
                preds = torch.zeros_like(val_labels)
                preds.scatter_(1, pred_idx, 1.0)
                inter = (preds * val_labels).sum(dim=(0, 2, 3, 4))
                pred_sum = preds.sum(dim=(0, 2, 3, 4))
                label_sum = val_labels.sum(dim=(0, 2, 3, 4))

                sum_inter += inter
                sum_pred += pred_sum
                sum_label += label_sum

        avg_val_loss = val_loss / val_step
        # FIX 3: eps 仅加在分母，避免分子分母不对称
        class_dice = (2.0 * sum_inter) / (sum_pred + sum_label + eps)
        mean_dice = class_dice[1:].mean().item()

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Dice Necrotic: {class_dice[1].item():.4f} | "
            f"Dice Edema: {class_dice[2].item():.4f} | "
            f"Dice Enhancing: {class_dice[3].item():.4f}"
        )

        dice_log = {
            f"val_dice_{name}": class_dice[idx].item()
            for idx, name in enumerate(class_names)
        }
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_dice_mean": mean_dice,
            "epoch": epoch + 1,
            **dice_log,
        })

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model_softmax.pth")
            print(f"  → 保存最优模型（val_loss={best_val_loss:.4f}）")

    wandb.finish()

def main():
    parser=argparse.ArgumentParser(description="Train SwinUNETR for brain tumor segmentation")
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()
    processed_dir = args.processed_dir
    plan_path=os.join(processed_dir, "Dataset101_Meningioma", "nnUNetPlans.json")
    path=os.join(processed_dir, "Dataset101_Meningioma")
    train(path, plan_path)

if __name__ == "__main__":
    main()