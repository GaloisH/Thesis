import glob
import monai
from plan2transform import build_transforms_from_plan
import torch
from monai.networks.nets import SwinUNETR
from monai.data import Dataset, DataLoader
import os
import wandb

wandb.init(
    project="swinunetr",
    name="swinunetr_100epochs",
    config={
        "epochs": 100,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "feature_size": 24,
        "drop_rate": 0.2,
        "attn_drop_rate": 0.0,
        "dropout_path_rate": 0.2,
        "use_checkpoint": True
    }
)

def train(path, plan_path):
    modality=["t1", "t1ce", "t2", "flair"]
    images_t1 = sorted(glob.glob(os.path.join(path, "imagesTr", "*0.nii.gz")))
    images_t1ce = sorted(glob.glob(os.path.join(path, "imagesTr", "*1.nii.gz")))
    images_t2 = sorted(glob.glob(os.path.join(path, "imagesTr", "*2.nii.gz")))
    images_flair = sorted(glob.glob(os.path.join(path, "imagesTr", "*3.nii.gz")))
    masks = sorted(glob.glob(os.path.join(path, "labelsTr", "*.nii.gz")))
    data_dicts={"image": [images_t1, images_t1ce, images_t2, images_flair],
                 "label": masks}
    
    
    train_transform, val_transform = build_transforms_from_plan(plan_path)
    
    # 使用 monai.data.Dataset 而不是 ImageDataset，并使用 collate_fn 避免多 crop 返回 list 报错
    from monai.data.utils import pad_list_data_collate
    train_ds = Dataset(data=data_dicts, transform=train_transform)
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=2, collate_fn=pad_list_data_collate
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 多分类时，通常 label 为 (B, 1, D, H, W)，需要 to_onehot_y 才能计算 DiceLoss
    loss_fn = monai.losses.DiceLoss(softmax=True, to_onehot_y=True)
    
    # 降低 feature_size 从 48 到 24 进一步节省显存
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=24,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    epochs = 50
    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            # 使用 scaler 缩放并反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(
                f"Epoch {epoch + 1}/{epochs}, Step {step}/{epoch_len}, Loss: {loss.item():.4f}"
            )
        epoch_loss /= step
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1} Average Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    plan_path = r"D:\python_code\projects\thesis\datasets\nnUNet_preprocessed\Dataset101_Meningioma\nnUNetPlans.json"
    path = r"D:\python_code\projects\thesis\datasets\nnUNet_raw\Dataset101_Meningioma"
    train(path, plan_path)

