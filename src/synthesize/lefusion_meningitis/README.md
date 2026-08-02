# 脑膜炎 LeFusion-H 使用说明

本目录把官方 LeFusion 的 3D U-Net 改造成单通道脑膜炎病灶合成流水线。数据增强和训练
Dataset 使用 MONAI 的字典式 Compose 管线。官方代码位于
`src/synthesize/LeFusion/`，本实现只导入其网络结构，不修改官方源码。

首版使用 `Dataset002_Meningitis` 的通道 0、`32³` patch、16-bin 病灶直方图条件和真实
掩膜轻度形变。输出是独立的单通道 nnUNet 数据集，不能与原三通道实验直接混用。

## 1. 环境与检查

项目指定解释器：

```powershell
D:\python_code\miniconda\python.exe
```

安装项目基础依赖和官方 LeFusion 依赖：

```powershell
D:\python_code\miniconda\python.exe -m pip install -r requirements.txt
D:\python_code\miniconda\python.exe -m pip install -r src/synthesize/LeFusion/requirements.txt
```

检查关键依赖和 GPU：

```powershell
D:\python_code\miniconda\python.exe -c "import monai, numpy, scipy, nibabel, yaml, torch; print(monai.__version__, torch.__version__, torch.cuda.is_available())"
D:\python_code\miniconda\python.exe -m pip check
```

实际扩散训练应使用 CUDA 版 PyTorch。CPU 可以执行数据准备和单元测试，但不适合完成
50,000 step 的 3D 扩散训练。

查看所有命令：

```powershell
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis --help
```

所有命令默认读取 `config/lefusion_meningitis.yaml`。使用其他配置：

```powershell
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis `
  --config config/lefusion_meningitis.yaml prepare
```

临时覆盖配置时，`--set` 必须放在子命令前，并可重复使用：

```powershell
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis `
  --set seed=123 `
  --set data.patch_size="[32,32,32]" `
  prepare
```

## 2. 推荐执行顺序

### 2.1 `prepare`：准备患者划分与病灶 patch

```powershell
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis prepare
```

功能：

- 扫描 `Dataset002_Meningitis/imagesTr` 的通道 0；
- 按病灶体积和连通分量统计生成 28/6/6 患者划分；
- 将影像转换为 RAS，并在脑区内做 z-score；
- 提取满足体积和边界要求的 `32³` 病灶 patch；
- 生成训练病灶直方图库和空间位置先验。

主要输出位于 `datasets/lefusion_meningitis/prepared/`：

```text
split.json               患者划分、病例统计和 split hash
manifest.json            patch 来源、裁剪坐标、归一化参数和 manifest hash
patches/*.npz            image、mask、histogram 三个数组
train_histograms.npy     只来自训练患者的直方图库
position_prior.json      只来自训练患者的位置先验
```

如果病例数不是配置中的 40 例，命令会拒绝生成划分，应先修改 `data.split_counts`。

### 2.2 `train`：训练 LeFusion-H

```powershell
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis train
```

训练使用：

- 官方 LeFusion `Unet3D`；
- 300 个扩散时间步；
- 16 维直方图条件；
- 病灶体素归一化 L1 噪声损失；
- AMP、梯度累积、EMA、梯度裁剪和验证早停。

默认输出位于 `outputs/lefusion_meningitis/training/`：

```text
best.pt                  验证损失最优的 EMA checkpoint
last.pt                  训练结束或早停时的 checkpoint
step_XXXXXX.pt           周期 checkpoint
history.json             train/validation loss 历史
```

从断点恢复：

```powershell
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis `
  --set training.resume="outputs/lefusion_meningitis/training/last.pt" `
  train
```

减小显存占用：

```powershell
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis `
  --set training.batch_size=1 `
  --set training.effective_batch_size=16 `
  train
```

### 2.3 `validate`：验证 checkpoint

```powershell
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis validate
```

该命令载入 `synthesis.checkpoint` 指定的 EMA 权重，在验证 patch 上计算前景归一化噪声
损失，并写出 `outputs/lefusion_meningitis/training/validation.json`。

验证其他 checkpoint：

```powershell
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis `
  --set synthesis.checkpoint="outputs/lefusion_meningitis/training/step_020000.pt" `
  validate
```

### 2.4 `synthesize`：生成完整影像—标签对

```powershell
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis synthesize
```

每个目标病例执行以下流程，并在同一个 RAS 空间状态中累计病灶：

1. 从训练患者选择真实病灶掩膜并做轻度旋转、缩放；
2. 根据训练位置先验在目标脑区寻找位置；
3. 避开已有病灶及其膨胀保护区；
4. 从训练直方图库采样纹理条件；
5. 在每个反向扩散步注入真实背景；
6. 硬合成病灶区域并执行 QC；成功后同步更新影像、总标签和累计插入 mask；
7. 后续病灶避开原始病灶及此前插入病灶；
8. 所有病灶处理结束后，只写出一份最终病例，并恢复原始方向、affine、spacing 和 header。

默认输出位于 `outputs/lefusion_meningitis/synthetic/`：

```text
images/*_0000.nii.gz     合成后的单通道影像
labels/*.nii.gz          原标签与新病灶掩膜的并集
masks/*.nii.gz           该病例累计插入的全部病灶掩膜
metadata/*.json          病例级结果以及每个病灶的来源、随机种子、尝试次数和 QC
summary.json             病例数、目标/成功病灶数、失败尝试数和完成率
```

令每个训练病例的最终合成影像累计两个新增病灶：

```powershell
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis `
  --set synthesis.num_per_case=2 `
  synthesize
```

### 2.5 `export`：导出独立 nnUNet 数据集

```powershell
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis export
```

默认创建 `Dataset202_MeningitisLeFusion`，其中包含训练划分的真实单通道病例、指定比例的
QC 合格合成病例，以及固定测试划分。`metadata/provenance.json` 保存全部数据来源。

输出目录非空时命令会停止，防止覆盖已有实验。确认替换配置指定的数据集时：

```powershell
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis export --force
```

为 S3/S4/S5 建立不同 Dataset ID：

```powershell
# S3：0.5 倍合成量
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis `
  --set export.synthetic_ratio=0.5 `
  --set export.output_dataset="datasets/nnUNet_raw/Dataset203_MeningitisLeFusion05x" `
  export

```

累计合成每个源病例最多产生一份合成病例，因此 `export.synthetic_ratio` 必须位于
`[0, 1]`。`synthesis.num_per_case` 控制单个最终影像中的新增病灶数，不再控制合成病例份数。

### 2.6 `evaluate`：汇总 QC 与分割指标

只汇总合成 QC：

```powershell
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis evaluate
```

同时评估 nnUNet 预测：

```powershell
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis `
  --set evaluation.predictions_dir="outputs/nnunet_predictions" `
  --set evaluation.references_dir="datasets/nnUNet_raw/Dataset202_MeningitisLeFusion/labelsTs" `
  evaluate
```

输出包括 Dice、HD95、NSD、病灶级召回率、每例假阳性病灶数，以及按参考标签体素数划分的
small/medium/large 层级。

## 3. 配置文件说明

`config/lefusion_meningitis.yaml` 各段用途：

| 配置段 | 用途 |
|---|---|
| `data` | 原始数据、通道、标签、patch、患者划分和准备目录 |
| `normalization` | z-score 裁剪尺度和直方图 bin 数 |
| `augmentation` | MONAI 同步空间增强及失败重试次数 |
| `model` | U-Net 输入、直方图条件和扩散时间步 |
| `training` | batch、学习率、AMP、EMA、早停和 checkpoint |
| `synthesis` | 权重、合成数量、掩膜变形、位置搜索和 QC |
| `export` | nnUNet Dataset ID、真实数据与合成比例 |
| `evaluation` | 预测/参考目录、NSD 容差和报告路径 |

正式五折实验必须为每一折使用独立的 `prepared_dir`、训练输出目录、合成输出目录和 Dataset
ID，并确保该折验证病例不参与直方图、位置先验、生成模型训练或 QC 阈值估计。

## 4. 各代码文件用途

| 文件 | 作用 | 常用入口 |
|---|---|---|
| `cli.py` | 解析命令行并分派七个阶段 | `python -m ... <command>` |
| `config.py` | YAML 读取、覆盖项合并和路径解析 | `load_config()` |
| `io.py` | nnUNet 命名、NIfTI/JSON 读写、方向恢复和哈希 | `load_ras_with_source()` |
| `data/` | 归一化、连通域、划分、patch 及 MONAI 增强/Dataset | `prepare()` |
| `losses.py` | 病灶前景归一化 L1/L2 损失 | `masked_foreground_loss()` |
| `model.py` | 官方 U-Net 包装、DDPM 前向过程和背景注入采样 | `LeFusionH()` |
| `training.py` | 训练、EMA、验证、早停和 checkpoint | `train()` / `validate()` |
| `synthesis/` | 掩膜变形、位置选择、累计合成、重试与 QC | `synthesize()` |
| `visualization/` | 固定 mask 推理、绘图以及 CSV/HTML 报告 | `visualize()` |
| `export.py` | 生成独立单通道 nnUNet 数据集 | `export_nnunet()` |
| `evaluation.py` | 合成 QC 和下游分割指标 | `evaluate()` |
| `__main__.py` | 支持 `python -m` 启动 | 无需直接调用 |

## 5. 在 Python 中调用

```python
from src.synthesize.lefusion_meningitis.config import load_config
from src.synthesize.lefusion_meningitis.data import prepare
from src.synthesize.lefusion_meningitis.training import train

cfg = load_config("config/lefusion_meningitis.yaml")
prepare_report = prepare(cfg)
train_report = train(cfg)
```

单独计算病灶聚焦损失：

```python
from src.synthesize.lefusion_meningitis.losses import masked_foreground_loss

loss = masked_foreground_loss(predicted_noise, true_noise, lesion_mask, "l1")
```

## 6. 测试

```powershell
D:\python_code\miniconda\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

关键验收项包括：患者划分互斥、固定 seed 可复现、归一化可逆、稀疏前景损失不被背景
稀释、掩膜外背景逐体素不变、标签并集正确和 NIfTI 几何信息保持。

