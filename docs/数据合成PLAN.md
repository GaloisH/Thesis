# 无条件 3D Latent Diffusion 病灶生成实验与代码方案

## 1. 方案摘要

采用两阶段生成模型：

1. 使用 `AutoencoderKL` 将 `32×32×32` 病灶 patch 压缩为 `4×8×8×8` latent；
2. 冻结 VAE，在 latent 空间训练无条件 `DiffusionModelUNet`；
3. 从高斯噪声随机采样 latent，经 VAE 解码得到病灶纹理和对应 mask；
4. 根据目标脑 MR 局部组织的强度统计反归一化病灶，再用软边界融合到原图；
5. 新标签为原标签与生成 mask 的并集。

默认采用“双通道联合生成”：

- 通道 0：相对于病灶周围组织的标准化强度残差；
- 通道 1：病灶二值 mask。

扩散模型仍然是完全无 condition 的，只接收加噪 latent 和时间步。双通道设计解决了单纯生成灰度 patch 后无法可靠获得融合区域和分割标签的问题。

实现基于 MONAI 1.4 的 `AutoencoderKL`、`DiffusionModelUNet`、`DDPMScheduler` 和 `LatentDiffusionInferer`。MONAI 官方 3D LDM 示例采用相同的两阶段训练方式，并强调应计算 latent scale factor。[MONAI 3D LDM 示例](https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/3d_ldm/3d_ldm_tutorial.py)、[LatentDiffusionInferer 文档](https://docs.monai.io/en/1.4.0/inferers.html)

## 2. 数据集与 Patch 构建

### 数据划分

默认使用 `Dataset002_Meningitis` 的通道 0：

- 训练：28 个患者；
- 验证：6 个患者；
- 测试：6 个患者；
- 按病灶总体积和连通分量数分层；
- 所有 patch 在患者划分完成后生成，禁止患者级泄漏。

训练集每例生成 32 个 patch，验证集和测试集每例固定生成 8 个 patch。预计得到约：

- 训练 patch：896 个；
- 验证 patch：48 个；
- 测试 patch：48 个。

如果某病例有效病灶不足，允许对同一病灶使用不同裁剪偏移和空间增强，但元数据中保留来源病例及连通分量编号。

### 病灶采样

1. 将影像和 mask 统一为 RAS 方向。
2. spacing 使用训练集的逐轴中位数。
3. 对 mask 做三维连通域分析。
4. 优先选择包围盒不超过 `24³` 的完整病灶，使其与 `32³` patch 边界至少相距 4 个体素。
5. 包围盒超过 `24³` 的大病灶不进入 MVP 生成集，避免训练出被 patch 截断的形状。
6. 裁剪中心为病灶质心加 `[-3,3]` 体素随机偏移。
7. 不足 `32³` 的图像边缘使用反射填充影像、零填充 mask。

训练 patch 可执行同步增强：

- 三轴翻转；
- ±15°旋转；
- `0.85-1.15` 倍缩放；
- 小幅弹性形变；
- gamma `0.8-1.2`；
- 高斯噪声标准差不超过局部 IQR 的 3%。

增强后的 mask 若接触 patch 边界、体积小于 8 体素或主体破碎，则重新采样。

### 局部残差表示

直接训练原始 MR patch 会让无条件模型同时学习大量正常背景，并在融合时产生灰度不匹配。因此将病灶转成相对于周围组织的残差。

定义病灶外侧 2-5 体素的环形区域为 \(S\)，计算：

\[
m_S=\operatorname{median}(I_S),\qquad
q_S=\operatorname{IQR}(I_S).
\]

若 \(q_S<10^{-6}\)，丢弃该 patch。病灶残差为：

\[
R=\operatorname{clip}
\left(
\frac{I-m_S}{q_S},-5,5
\right)/5.
\]

只保留病灶及向外膨胀 1 个体素的区域，其他位置设为 0。最终 VAE 输入为：

\[
X=\operatorname{concat}(R,M),
\qquad X\in\mathbb{R}^{2\times32\times32\times32}.
\]

每个 patch 保存：

- 双通道数组；
- 原始病灶 mask；
- `m_S`、`q_S`；
- 原病例、连通分量和裁剪坐标；
- spacing、方向及增强参数。

## 3. AutoencoderKL 实验

### 网络配置

```text
spatial_dims      = 3
in_channels       = 2
out_channels      = 2
num_channels      = (32, 64, 128)
latent_channels   = 4
num_res_blocks    = 2
norm_num_groups   = 16
attention_levels  = (False, False, True)
```

两次下采样后：

```text
输入：  2 × 32 × 32 × 32
latent：4 × 8 × 8 × 8
输出：  2 × 32 × 32 × 32
```

输出通道 0 直接作为残差重建值，通道 1 作为 mask logits，经 sigmoid 后得到软 mask。

### VAE 损失

图像残差损失：

\[
L_{\mathrm{img}}
=
\operatorname{mean}
\left[
(1+4M)|\hat R-R|
\right].
\]

mask 损失：

\[
L_{\mathrm{mask}}
=
L_{\mathrm{BCE}}(\hat M,M)
+
L_{\mathrm{Dice}}(\hat M,M).
\]

KL 损失：

\[
L_{\mathrm{KL}}
=
\frac{1}{2}
\sum
\left(
\mu^2+\sigma^2-\log \sigma^2-1
\right).
\]

总损失：

\[
L_{\mathrm{VAE}}
=
L_{\mathrm{img}}
+0.5L_{\mathrm{mask}}
+10^{-6}L_{\mathrm{KL}}.
\]

主实验不启用 PatchGAN，避免在不足 1,000 个训练 patch 上出现判别器过拟合。另设一个消融实验，在 VAE 训练 20 epoch 后加入：

- 2.5D perceptual loss，权重 `0.001`；
- PatchGAN adversarial loss，权重 `0.01`。

只有当该消融模型在验证集上改善病灶纹理误差且不降低 mask Dice 时，才用于后续 LDM。

### 训练参数

```text
optimizer          = AdamW
learning_rate      = 1e-4
weight_decay       = 1e-5
batch_size         = 8（单卡32GB）
batch_size_per_gpu = 4（双卡16GB）
epochs             = 300
warmup_epochs      = 5
scheduler          = cosine
AMP                = true
gradient_clip      = 1.0
early_stop         = 30 epochs
```

最佳模型按以下综合分数选择：

\[
S_{\mathrm{VAE}}
=
L_{\mathrm{img,val}}
+0.5(1-\mathrm{Dice}_{\mathrm{mask,val}}).
\]

进入 LDM 阶段的最低要求：

- 验证集 mask Dice ≥ 0.90；
- 病灶区域归一化 MAE ≤ 0.10；
- 95% 以上重建 mask 不接触 patch 边界；
- 无持续 KL 爆炸或 posterior collapse。

## 4. Latent 缓存与无条件 LDM

### Latent 缓存

VAE 训练结束后冻结全部参数。对每个 patch 保存 posterior 的：

- `z_mu`；
- `z_sigma`；
- patch ID 和患者 ID。

LDM 训练时采样：

\[
z=z_\mu+z_\sigma\epsilon,\qquad
\epsilon\sim\mathcal N(0,I).
\]

scale factor 必须根据完整训练集 latent 的运行方差计算，而不是只使用一个 batch：

\[
s=\frac{1}{\operatorname{Std}(z)}.
\]

训练扩散模型时使用 \(z_s=s z\)，采样解码前再除以 \(s\)。scale factor 与 VAE checkpoint 一起固化。

### DiffusionModelUNet

```text
spatial_dims       = 3
in_channels        = 4
out_channels       = 4
num_channels       = (64, 128, 256)
num_res_blocks     = 2
attention_levels   = (False, True, True)
num_head_channels  = (0, 64, 64)
condition           = None
```

模型输入只有：

- 加噪 latent；
- diffusion timestep。

不得传入 mask、位置、病例或强度条件，确保实验确实为 unconditional latent diffusion。

### 扩散配置

```text
num_train_timesteps = 1000
schedule            = scaled_linear_beta
beta_start          = 0.0015
beta_end            = 0.0195
prediction_type     = epsilon
loss                = MSE(noise_pred, noise)
```

以上 scheduler 参数与 MONAI 官方 3D LDM 示例保持一致。[官方配置](https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/3d_ldm/3d_ldm_tutorial.py)

训练参数：

```text
optimizer          = AdamW
learning_rate      = 1e-4
weight_decay       = 1e-4
effective_batch    = 32
max_steps          = 50000
warmup_steps       = 1000
scheduler          = cosine
EMA                = 0.9999
AMP                = true
gradient_clip      = 1.0
validation_interval = 2500 steps
```

单卡 32GB 使用 batch 32；双卡 16GB 使用每卡 batch 16。若显存不足，降为每卡 8 并通过梯度累积保持 effective batch 32。

采样设置：

- 质量基准：DDPM 1,000 步；
- 批量生成：DDIM 50 步；
- 每次固定保存 seed；
- 默认生成候选数量为目标数量的 1.5 倍，用 QC 筛选后补采样。

## 5. 解码、Mask 后处理与质量控制

解码结果为：

\[
(\hat R,\hat M)=D(z/s).
\]

mask 后处理：

1. `sigmoid(mask_logits) > 0.5`；
2. 去除小于 8 体素的连通分量；
3. 默认保留最大连通分量；
4. 填充内部小孔；
5. 进行一次闭运算；
6. 拒绝接触 `32³` patch 边界的样本。

候选样本还必须满足：

- mask 体积位于真实训练病灶第 2.5-97.5 百分位；
- 三轴长度和紧致度位于训练分布合理范围；
- 残差中至少 90% 的非零能量位于 mask 或其 1 体素边界内；
- 无 NaN、Inf 或整块常量输出；
- mask 概率均值不处于 `0.1-0.9` 的模糊失败区间；
- 最多重采样 20 次，仍失败则记录并跳过。

通过 QC 的病灶保存为独立素材：

```text
residual: float32 [32,32,32]
mask:     uint8   [32,32,32]
seed
vae_checkpoint
ldm_checkpoint
scale_factor
qc_metrics
```

## 6. 融合到脑 MR

### 位置采样

默认向现有病例的正常区域添加病灶：

1. 根据训练病例病灶质心的归一化坐标分布采样候选中心；
2. 映射到目标病例尺寸；
3. 候选 `32³` patch 必须完整位于图像内；
4. 不得与原病灶膨胀 5 mm 后的区域重叠；
5. 目标局部非零体素比例应大于 80%；
6. 目标环形区域 IQR 必须大于 `10^-6`；
7. 目标局部强度统计应位于训练病灶邻域统计的第 5-95 百分位。

### 反归一化

在目标位置计算环形背景：

\[
m_T=\operatorname{median}(I_T),\qquad
q_T=\operatorname{IQR}(I_T).
\]

生成病灶强度为：

\[
I_G=m_T+5q_T\hat R.
\]

将 \(I_G\) 裁剪到目标病例非零脑区强度的第 0.5-99.5 百分位。

### 软融合

根据生成 mask 的有符号距离变换构造 2 体素过渡带：

- mask 内部超过 2 体素：\(\alpha=1\)；
- mask 外部：\(\alpha=0\)；
- 边界带：使用余弦函数从 0 平滑过渡到 1。

融合结果：

\[
I_{\mathrm{new}}
=
(1-\alpha)I_{\mathrm{target}}
+
\alpha I_G.
\]

标签更新为：

$$
M_{\mathrm{new}}
=
M_{\mathrm{original}}\lor M_{\mathrm{generated}}.
$$

mask 外的图像必须逐体素保持原值。输出 NIfTI 继承目标影像的 affine、spacing、方向和 header。

## 7. 实验矩阵

### 生成质量实验

| 实验 | Patch 表示 | VAE | 生成方式 | 融合 |
|---|---|---|---|---|
| E0 | 真实病灶 | 无 | Copy-Paste + 形变 | 软融合 |
| E1 | 原始灰度 patch | AutoencoderKL | 无条件 LDM | 直方图匹配 |
| E2 | 残差 + mask | AutoencoderKL | 无条件 LDM | 软融合 |
| E3 | 残差 + mask | AutoencoderKL + perceptual/GAN | 无条件 LDM | 软融合 |

主要比较 E2 与 E0；E1 用于验证局部残差表示的必要性；E3 验证对抗与感知损失是否值得保留。

生成质量指标：

- VAE 重建 MAE、PSNR、SSIM、mask Dice；
- 真实与生成病灶的体积、三轴长度、紧致度分布；
- 病灶强度均值、标准差、偏度、峰度；
- GLCM 纹理特征和 MMD；
- 生成样本两两 MS-SSIM；
- 融合边界梯度跳变；
- QC 通过率和平均重采样次数。

### 下游分割实验

使用完全一致的 nnU-Net 划分、预处理和训练参数：

| 组别 | 训练数据 |
|---|---|
| S0 | 仅真实数据 |
| S1 | 真实数据 + Copy-Paste |
| S2 | 真实数据 + LDM 合成 0.5× |
| S3 | 真实数据 + LDM 合成 1× |
| S4 | 真实数据 + LDM 合成 2× |

报告：

- Dice；
- HD95；
- NSD；
- 病灶级召回率；
- 每例假阳性数；
- 按病灶体积分层的小、中、大病灶结果。

最终采用 5 折患者级交叉验证。每一折都必须独立构建 patch 数据、训练 VAE/LDM 或至少独立限制生成素材来源，测试折不得参与任何生成模型训练和 QC 分布估计。

成功标准：

- S3 相对 S0 的平均 Dice 提高至少 1 个百分点，或病灶级召回率提高至少 3 个百分点；
- HD95 不恶化超过 10%；
- mask 外背景差异严格为 0；
- QC 通过率达到 80%；
- E2 的边界梯度跳变显著低于 E1。

## 8. 代码组织与接口

新增 `src/lesion_ldm/` 包，内部按功能组织：

```text
data/       Patch提取、残差构建、split、latent缓存
models/     AutoencoderKL与DiffusionModelUNet构造
training/   VAE训练、LDM训练、checkpoint和EMA
sampling/   latent采样、解码、mask后处理、QC
fusion/     位置采样、反归一化、软融合、NIfTI写回
metrics/    重建、生成质量和下游指标
cli.py      统一命令入口
```

统一配置文件为 `config/lesion_ldm.yaml`，测试放在 `tests/lesion_ldm/`。

CLI：

```bash
python -m src.lesion_ldm.cli prepare-patches --config config/lesion_ldm.yaml
python -m src.lesion_ldm.cli train-vae       --config config/lesion_ldm.yaml
python -m src.lesion_ldm.cli cache-latents   --config config/lesion_ldm.yaml
python -m src.lesion_ldm.cli train-ldm       --config config/lesion_ldm.yaml
python -m src.lesion_ldm.cli sample          --config config/lesion_ldm.yaml --num 500
python -m src.lesion_ldm.cli fuse            --config config/lesion_ldm.yaml --num-per-case 1
python -m src.lesion_ldm.cli evaluate        --config config/lesion_ldm.yaml
```

checkpoint 必须保存：

- 模型、优化器、scheduler、GradScaler 和 EMA；
- epoch、global step；
- 完整配置；
- 数据 split 和 patch manifest 哈希；
- latent scale factor；
- Python、NumPy、PyTorch 和 CUDA RNG 状态。

合成数据写入新的 nnUNet 数据集，不覆盖原始数据：

```text
imagesTr/{case_id}_synNN_0000.nii.gz
labelsTr/{case_id}_synNN.nii.gz
metadata/{case_id}_synNN.json
```

## 9. 测试要求与默认假设

测试覆盖：

- `32³` 裁剪、边缘填充和坐标逆映射；
- 图像与 mask 的同步增强；
- 残差归一化和反归一化互逆；
- AutoencoderKL 输入、latent 和输出尺寸；
- scale factor 的完整训练集计算；
- 无 condition 的 LDM 前向与采样；
- mask 后处理和拒绝采样；
- 融合后 affine/header 不变；
- alpha 区域外图像严格不变；
- 标签并集正确；
- 固定 seed 可重复；
- DDP checkpoint 可由单卡加载。

默认假设：

- 使用 `Dataset002_Meningitis` 通道 0；
- 生成纹理和 mask 的双通道 patch；
- 主要向阳性病例的正常区域增加病灶；
- 只处理能够完整容纳于 `32³` patch 的中小病灶；
- 第一阶段不生成多模态 MRI、不模拟占位效应，也不使用位置或类别 condition；
- 主模型不使用对抗损失，PatchGAN 仅作为独立消融实验。
