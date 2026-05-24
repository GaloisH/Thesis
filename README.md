# Meningioma Segmentation & Registration Pipeline

多模态 MRI 脑膜瘤分割与配准分析 pipeline，基于 nnUNet + SwinUNETR 集成模型进行肿瘤分割，并通过 ANTs SyN 配准到 MNI 标准空间进行病灶概率统计与脑区分析。

## Pipeline 概览

```
原始 BraTS 数据
     │
     ▼
① prepare_data.py     → 转换 nnUNet 格式 (Dataset101_Meningioma)
     │
     ▼
② train.py            → nnUNet 训练 (2D, fold 0)
   swin_train.py      → SwinUNETR 训练 (MONAI)
     │
     ▼
③ nnunet_predict.py   → nnUNet 推理 (softmax 概率图)
   swin_predict.py    → SwinUNETR 推理 (softmax 概率图)
     │
     ▼
④ ensemble_predict.py → 加权融合 (默认 0.7×nnUNet + 0.3×SwinUNETR)
     │
     ▼
⑤ registration.py     → ANTs SyN 配准到 MNI152NLin2009cAsym
     │
     ▼
⑥ visualization.py    → 概率图可视化 + AAL3 脑区统计
```

## 目录结构

```
thesis/
├── src/
│   ├── segmentation/          # 分割模块
│   │   ├── prepare_data.py    # 数据预处理 (BraTS → nnUNet 格式)
│   │   ├── train.py           # nnUNet 训练启动脚本
│   │   ├── swin_train.py      # SwinUNETR 训练 (MONAI)
│   │   └── plan2transform.py  # nnUNetPlans → MONAI transforms
│   │
│   ├── inference/             # 推理模块
│   │   ├── nnunet_predict.py  # nnUNet 推理
│   │   ├── swin_predict.py    # SwinUNETR 推理
│   │   ├── ensemble_predict.py # 加权集成融合
│   │   └── compare_models.py  # 模型对比可视化 (Dice + 切片)
│   │
│   └── registration/          # 配准模块
│       ├── registration.py    # CLI 入口
│       ├── registration_core.py # 核心引擎 (ANTs SyN, 并行, 断点续跑)
│       ├── visualization.py   # 可视化 (切片, 叠加, 直方图, 脑区统计)
│       └── atlas_ready/       # AAL3v2 脑图谱 (170 区域)
│
├── datasets/                  # 数据与模型 (gitignored)
│   ├── raw/                   # 原始 BraTS2020 数据
│   ├── nnUNet_raw/Dataset101_Meningioma/  # nnUNet 格式数据
│   │   ├── imagesTr/          # 训练图像 (368 例 × 4 模态)
│   │   ├── labelsTr/          # 训练标签
│   │   └── imagesTs/          # 测试图像 (125 例)
│   ├── nnUNet_preprocessed/   # nnUNet 预处理结果
│   └── nnUNet_results/        # 训练好的 nnUNet 模型
│
└── datasetsregistration/      # 配准输出 (gitignored)
    ├── registered/            # 每例 MNI 空间配准结果
    ├── visualizations/        # 分析可视化
    ├── lesion_probability_map.nii.gz
    ├── lesion_frequency_map.nii.gz
    └── region_probability_by_atlas.csv
```

## 数据格式

### 输入影像

每例包含 4 个模态，统一命名为 `{case_id}_{channel:04d}.nii.gz`：

| 通道 | 模态 | 文件名后缀 |
|------|------|-----------|
| 0 | T1 | `_0000.nii.gz` |
| 1 | T1ce (增强 T1) | `_0001.nii.gz` |
| 2 | T2 | `_0002.nii.gz` |
| 3 | FLAIR | `_0003.nii.gz` |

### 分割标签

单文件 `{case_id}.nii.gz`，整数标签：

| 值 | 含义 |
|----|------|
| 0 | 背景 (background) |
| 1 | 坏死核心 (necrotic) |
| 2 | 瘤周水肿 (edema) |
| 3 | 增强肿瘤 (enhancing tumor) |

## 各阶段详解

### ① 数据预处理 (`src/segmentation/prepare_data.py`)

将原始 BraTS2020 数据（按病例分目录存放）转换为 nnUNetv2 所需的标准格式。

### ② 模型训练

**nnUNet**（2D U-Net，5 折交叉验证）：

```bash
cd src/segmentation
python train.py
```

**SwinUNETR**（3D Swin Transformer，MONAI 框架）：

```bash
cd src/segmentation
python swin_train.py
```

训练配置：100 epochs, lr=1e-4, feature_size=48, dropout=0.2, 滑动窗口推理 (overlap=0.5)。

### ③ 推理

**nnUNet 推理** — 输出 softmax 概率图 (`.npy`) 和硬分割 (`.nii.gz`)：

```bash
cd src/inference
python nnunet_predict.py \
    -i t1.nii.gz t1ce.nii.gz t2.nii.gz flair.nii.gz \
    -o ./output \
    -m ../../datasets/nnUNet_results/Dataset101_Meningioma/nnUNetTrainer__nnUNetPlans__2d \
    --device cuda
```

**SwinUNETR 推理** — 输出 softmax 概率图 (`.nii.gz`) 和硬分割 (`.nii.gz`)：

```bash
cd src/inference
python swin_predict.py
```

### ④ 集成融合 (`src/inference/ensemble_predict.py`)

对 nnUNet 和 SwinUNETR 的 softmax 概率图做加权平均（默认权重 0.7 / 0.3），生成集成分割结果：

```bash
cd src/inference
python ensemble_predict.py \
    --nnunet_dir /path/to/nnunet/prob \
    --swin_dir /path/to/swinunetr/prob \
    --raw_dir ../../datasets/nnUNet_raw/Dataset101_Meningioma/imagesTs \
    --output_dir ../../outputs/ensemble \
    --w_nnunet 0.7 --w_swin 0.3
```

### ⑤ 配准 (`src/registration/`)

将分割掩码通过 ANTs SyN 配准到 MNI152NLin2009cAsym 标准空间，构建病灶概率/频率图：

```bash
cd src/registration
python registration.py \
    --max-cases 100 \
    --output-dir ../../datasetsregistration \
    --atlas-path ./atlas_ready/AAL3v2.nii.gz \
    --atlas-labels ./atlas_ready/AAL3v2.json
```

**关键参数：**
- 配准算法：ANTs SyN (Symmetric Normalization)，4 层迭代 (100, 70, 50, 20)
- 模板：MNI152NLin2009cAsym (2mm, 去颅骨) via TemplateFlow
- 并行：ProcessPoolExecutor，内存感知 (~4GB/worker)
- 支持断点续跑 (`checkpoint.json`)

### ⑥ 可视化与分析 (`src/registration/visualization.py`)

生成三类分析图：

| 输出 | 内容 |
|------|------|
| `01_frequency.png` | 病灶频率图正交切片 |
| `02_probability.png` | 病灶概率图正交切片 |
| `03_overlay.png` | 概率图叠加到 MNI 模板 |
| `04_histogram.png` | 概率值分布直方图 |
| `05_regions.png` / `07_*.png` / `08_*.png` | AAL3 脑区风险排名 |
| `qc_{case_id}.png` | 单病例配准质控 |

```bash
cd src/registration
python visualization.py \
    --output-dir ../../datasetsregistration \
    --viz-dir ../../datasetsregistration/visualizations
```

### 模型对比 (`src/inference/compare_models.py`)

在训练集上对比 nnUNet、SwinUNETR、集成模型的 Dice 分数和 2D 切片：

```bash
cd src/inference
python compare_models.py \
    --images_dir ../../datasets/nnUNet_raw/Dataset101_Meningioma/imagesTr \
    --labels_dir ../../datasets/nnUNet_raw/Dataset101_Meningioma/labelsTr \
    --nnunet_model_dir ../../datasets/nnUNet_results/Dataset101_Meningioma/nnUNetTrainer__nnUNetPlans__2d \
    --swin_checkpoint /path/to/best_model.pth \
    --plan_path ../../datasets/nnUNet_preprocessed/Dataset101_Meningioma/nnUNetPlans.json \
    --output_dir ../../outputs/model_comparison \
    --max_cases 10
```

输出：箱线图 (Dice 分布) + 3×5 切片对比网格 + per-case CSV。

## 依赖

核心依赖：`torch`, `nnunetv2`, `monai`, `ants` (ANTsPy), `nibabel`, `templateflow`, `matplotlib`

完整安装：

```bash
pip install torch torchvision nnunetv2 monai antspyx nibabel templateflow matplotlib wandb
```
