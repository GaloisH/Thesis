# LeFusion 仓库作用与 DDPM 接口复用评估

> 审查日期：2026-08-22  
> 结论：LeFusion 仓库当前**有实际运行时作用**，但只复用了 `Unet3D` 及其基础块；现有 `GaussianDiffusion_Nolatent` 可作为重构基础，不能直接替换当前 DDPM 包装器。

## 1. 当前实际调用关系

```text
lefusion_meningitis/model.py
  -> load_official_unet()
  -> 将 src/synthesize/LeFusion/LeFusion 插入 sys.path
  -> from ddpm.diffusion import Unet3D
  -> LeFusionH.denoiser = Unet3D(...)
```

因此删除、漏拉取或移动 `src/synthesize/LeFusion` 后，`LeFusionH` 将无法构建。当前主项目把该路径记录为 Git mode `160000` 的 gitlink，指向 LeFusion commit：

```text
03dc67bd8169ced5f8bb6a8707d73377f110ebff
```

但工作区根目录未发现 `.gitmodules`。这会导致新 clone 不知道子模块 URL，标准 `git submodule update --init` 无法可靠恢复源码。当前本地 LeFusion 远程是 `https://github.com/HINTLab/LeFusion.git`。

### 被当前工程使用的 LeFusion 代码

- `ddpm.diffusion.Unet3D`
- `RelativePositionBias`、`Residual`、`SinusoidalPosEmb`
- `LayerNorm`、`PreNorm`、`Block`、`ResnetBlock`
- `SpatialLinearAttention`、`Attention`、`EinopsToAndFrom`
- `Downsample`、`Upsample`
- `ddpm.text.BERT_MODEL_DIM` 被模块导入，但当前 `use_bert_text_cond=False`，不会执行 BERT 条件路径。

### 存在但未被当前工程调用的 LeFusion 代码

- `GaussianDiffusion_Nolatent`
- vendored `Trainer` 和 EMA
- vendored Hydra 训练入口
- vendored inference/RePaint 入口
- LIDC/EMIDEC dataset 路径
- DiffMask 工程

## 2. 当前实例化的模型

`config/lefusion_meningitis.yaml` 给出：

```text
patch/image/depth : 32 x 32 x 32
channels          : 1
base dim          : 32  (config key 仍命名为 image_size)
histogram dim     : 16
dim_mults         : [1, 2, 4, 8]
timesteps         : 300
loss              : foreground-normalized L1 noise loss
```

实际构建验证结果：

```text
总参数量     : 10,033,825
可训练参数 : 10,033,809
U-Net 通道   : 32 -> 32 -> 64 -> 128 -> 256
平面分辨率   : 32 -> 16 -> 8 -> 4
深度 F        : 全程保持 32
```

## 3. 能否直接使用现有 `GaussianDiffusion_Nolatent`

**不能直接替换。**可以复用数学部分和 RePaint jump schedule，但需要先做针对脑膜炎任务的接口重构。

| 项目 | 当前 `LeFusionH` | vendored `GaussianDiffusion_Nolatent` | 直接替换后果 |
|---|---|---|---|
| 调度 | 线性 beta `1e-4 -> 2e-2` | cosine beta | 训练分布和 DDPM buffers 不兼容 |
| `T=300` 末态 | `alpha_bar=0.0480584` | `alpha_bar=2.70e-9` | 当前线性前向末态仍保留约 `sqrt(alpha_bar)=0.219` 的原始信号；cosine 才接近纯噪声 |
| 训练调用 | `model(image, mask, histogram)` | 要求 `diffusion(x=(image,hist), mask=mask)` | 参数位置不兼容 |
| 任务类型 | 通用 binary lesion mask | `data_type` 只实现 `lidc`/`emidec` | `meningitis` 路径中 `loss`/`mask` 可未定义 |
| 损失 | 每样本除以前景权重和 | mask 后对全张 patch 做 `F.l1_loss` | 损失随病灶体积比例缩放，训练目标改变 |
| 条件 | 直接传 `[B,16]` histogram | 训练时必须将 histogram 打包在 tuple `x` 中 | 需要 adapter |
| 普通采样 | `sample_patch(background,mask,hist,generator)` | `p_sample_loop(shape,cond,cond_scale)` | 官方普通采样不注入真实背景 |
| RePaint | 简化的每步背景注入 | 具备 jump schedule | 官方路径需 Hydra/OmegaConf 风格 `conf` 和 `model_kwargs` |
| mask 语义 | `True` 为生成区 | LIDC/EMIDEC 中通过特定 label ID 反转成 keep mask | 脑膜炎 label 语义需重写 |
| 随机数 | `sample_patch` 接收局部 `torch.Generator` | 内部使用全局 `torch.randn_like` | 现有按病灶 seed 复现契约丢失 |
| checkpoint | `denoiser.*` + 当前 DDPM buffers，`ema_model` | `denoise_fn.*` + 更多 buffers，vendored Trainer 用 `ema` | 无法 strict load，key 和 buffer 形状都不同 |
| AMP/训练器 | 项目自定义 AdamW、梯度累积、EMA、early stop | vendored Adam、DataParallel、自定义 Trainer | 训练行为和 checkpoint schema 改变 |

其他确定的接口问题：

- vendored 训练入口显式拒绝 `lidc`/`emidec` 以外的 `data_type`。
- `p_sample_repaint()` 也只为 LIDC label `1` 和 EMIDEC label `3/4` 构造 mask。
- `inpa_inj_sched_prev_cumnoise=True` 时会调用未在该类中定义的 `get_gt_noised()`。
- vendored inference 加载的是整个 diffusion state dict，而当前 checkpoint 的模型层级不同。

## 4. 推荐方案

### 短期：保留当前 `LeFusionH`（推荐）

这是对现有训练、EMA checkpoint、前景归一化损失、合成管线和按病灶 seed 复现契约影响最小的方案。

应优先：

1. 修复/补齐 `.gitmodules`，或将需要的 U-Net 以明确 license 和唯一包名纳入项目。
2. 去除全局 `sys.path` 注入，让导入路径可验证。
3. 保留已训 checkpoint 时不更改 schedule；对新训练实验单独评估 cosine schedule。
4. 记录 schedule 类型和完整模型配置到 checkpoint 中，加载时校验。

### 中期：提取一个通用 DDPM 内核

可以以 `GaussianDiffusion_Nolatent` 为参考，复用：

- `q_sample`、`predict_start_from_noise`、`q_posterior`、`p_mean_variance`
- cosine schedule 和 dynamic threshold
- `forward_with_cond_scale` 路径
- RePaint `get_schedule_jump()` 和 `undo()` 思路

但应新建通用接口，而不是让脑膜炎任务伪装成 `data_type="lidc"`：

```python
loss = diffusion.training_loss(
    clean=image,
    generate_mask=mask,
    condition=histogram,
    timestep=timestep,
    noise=noise,
)

patch = diffusion.inpaint(
    background=background,
    generate_mask=mask,
    condition=histogram,
    generator=generator,
    jump_schedule=optional_schedule,
)
```

接口内部应把以下策略参数化：

- `schedule = linear | cosine`
- `objective = pred_noise | pred_x0 | pred_v`
- `loss_reduction = foreground_mean | whole_patch_mean`
- `mask_semantics = generate_true`
- `clip_denoised` 与 `data_range`
- `jump_schedule`
- `cond_scale`

如果 schedule、目标或条件通道发生改变，应新建实验并重新训练，不应尝试把旧 checkpoint 强行视为等价模型。

## 5. 完整合成路径中的其他发现

- `position_prior.json` 在数据准备中生成，合成时也被读取到变量 `prior`，但当前后续没有使用它。实际位置先验已切换为 FastSurfer cortical candidate 均匀采样。
- 当前数据归一化确实将裁剪后 z-score 映射到 `[-1,1]`，因此模型反向时的 `clamp(-1,1)` 与当前数据管线一致。
- 采样后会额外执行 `brighten_lesion_interior()`；最终强度不是纯 DDPM 输出，而是 DDPM 结果加一个基于外环带统计的确定性内部增亮。
- `qc_patch()` 当前只拒绝 non-finite 结果；配置中 `max_boundary_jump_z` 和 `intensity_z_limit` 没有进入当前 QC 判定。
- FastSurfer 位置约束只要求**病灶中心**属于 `ctx-lh-*`/`ctx-rh-*` 体素，并不要求整个病灶 mask 都在皮层内。
