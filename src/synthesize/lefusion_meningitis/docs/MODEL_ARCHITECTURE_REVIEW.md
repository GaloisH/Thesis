# LeFusion-H 模型架构与代码审查

> 审查日期：2026-08-22  
> 主入口：`src/synthesize/lefusion_meningitis/model.py`

## 1. 审查边界

本文严格只使用 `model.py` 及理解该文件所必需的引用代码：

- `lefusion_meningitis/model.py`：模型工厂、DDPM 系数、训练前向、反向采样和 checkpoint 加载。
- `lefusion_meningitis/losses.py::masked_foreground_loss`：`forward()` 直接引用的损失函数。
- `lefusion_meningitis/logger.py::get_logger`：模型模块导入时直接引用的日志辅助函数。
- `LeFusion/LeFusion/ddpm/diffusion.py::Unet3D` 及其构建/前向依赖的基础块。
- `LeFusion/LeFusion/ddpm/text.py::BERT_MODEL_DIM`：`Unet3D` 导入的常量；当前包装器未启用 BERT 文本条件。

未查看或审查训练脚本、数据集、数据预处理、配置文件、调用方或其他文档。因此，本文不假定当前运行时配置的具体数值，也不对数据管线是否满足模型假设作出结论。

## 2. 结论摘要

当前模型是一个**直方图条件的 DDPM 病灶合成模型**，去噪器为 vendored LeFusion 的 `Unet3D`。它的“3D”不是常规的各向同性 3D CNN：卷积和空间上/下采样只在 `H×W` 平面运行，切片/帧维 `F` 不被卷积或采样，跨切片信息交互主要由 temporal attention 完成。更准确的描述是**2D 卷积 U-Net + 切片内空间注意力 + 跨切片时序注意力**。

必须特别注意：

1. **mask 不是去噪 U-Net 的输入条件。**训练时它只给损失加权，采样时它只控制哪些体素保留生成结果/哪些体素注入背景。U-Net 本身只接收带噪影像、时间步和直方图。
2. `config["image_size"]` 被传给 `Unet3D(dim=...)`，而 vendored 模型中 `dim` 的真实语义是**基础特征通道数**，不是空间尺寸。这是当前代码中最需要核对的配置语义。
3. `sample_patch()` 是每步注入带噪真实背景的 mask-clamped DDPM，但没有 RePaint 的 jump/resampling 时间表，因此注释中的 “RePaint-style” 只能理解为简化变体。
4. checkpoint 使用 `torch.load(..., weights_only=False)`，对不可信 checkpoint 存在 Python pickle 任意代码执行风险。

## 3. 符号与输入约定

| 符号 | 来源 | 含义 |
|---|---|---|
| `B` | 输入 | batch size |
| `C` | `config["channels"]` | 输入和输出影像通道数 |
| `F` | 输入 | 切片/帧/深度数，全网络不做上下采样 |
| `H, W` | 输入 | 平面空间尺寸 |
| `D` | `config["image_size"]` | 实际作为 U-Net 基础通道数 `dim` |
| `M=(m₁,…,mₗ)` | `config["dim_mults"]` | 各层通道倍数 |
| `K` | `config["histogram_dim"]` | 直方图条件向量长度 |
| `T` | `config["timesteps"]` | DDPM 总时间步数 |

预期主张量形状：

```text
image/background/noisy/predicted_noise : [B, C, F, H, W]
mask                                    : 训练可为 [B, 1, F, H, W]
                                          或 [B, C, F, H, W]
                                          采样必须与 background 完全同形
histogram                               : [B, K]
timestep                                : [B]，应为 torch.long
loss                                    : 标量
```

`model.py` 本身没有对上述大部分契约做显式验证。

## 4. 整体数据流

### 4.1 训练前向

```text
image x₀ [B,C,F,H,W]
  + 采样 t ~ Uniform{0,…,T-1}
  + 噪声 ε ~ N(0,I)
            |
            v
q_sample: xₜ = sqrt(ᾱₜ) x₀ + sqrt(1-ᾱₜ) ε
            |
            v
Unet3D(xₜ, t, histogram) -> εθ
            |
            v
masked_foreground_loss(εθ, ε, mask)
            |
            v
按每个样本的 mask 权重和归一化，再对 batch 取均值
```

损失为：

```text
L_b = Σ_i mask[b,i] * error(εθ[b,i] - ε[b,i]) / Σ_i mask[b,i]
L   = mean_b(L_b)
```

`error` 由 `loss_type` 决定：`l1` 为绝对值，`l2` 为平方。这种写法不会让大病灶样本仅因前景体素更多而在 batch 中占更大权重。

### 4.2 反向扩散

`_p_mean_variance()` 执行标准的噪声预测参数化：

```text
εθ       = Unet3D(xₜ, t, histogram)
x̂₀       = (xₜ - sqrt(1-ᾱₜ) εθ) / sqrt(ᾱₜ)
x̂₀       = clamp(x̂₀, -1, 1)
μθ       = coef1[t] * x̂₀ + coef2[t] * xₜ
σ²       = posterior_variance[t]
```

beta 采用固定线性时间表：

```text
βₜ = linspace(1e-4, 2e-2, T)
αₜ = 1 - βₜ
ᾱₜ = ∏ₛ₌₀…ₜ αₛ
```

相关系数均通过 `register_buffer()` 保存，会跟随模型设备迁移并进入 `state_dict`。

### 4.3 `sample_patch()` 的区域约束采样

从全高斯噪声开始，对 `t=T-1…0` 逐步执行：

1. 用当前 `t` 对真实 `background` 做前向加噪。
2. 在 mask 外用带噪背景替换当前样本，mask 内保留待生成状态。
3. 调用 U-Net 得到反向后验均值与方差。
4. `t>0` 时加入新高斯噪声；`t=0` 时只取均值。
5. 循环结束后，mask 外强制返回原始背景。

这保证最终 mask 外的体素与输入背景逐值一致，但不保证边界内外的结构/强度连续性。

## 5. U-Net 详细架构

### 5.1 通道与尺度

`Unet3D` 得到：

```text
dims = [D, D*m₁, D*m₂, ..., D*mₗ]
```

包装器不传 `init_dim`，所以 `init_dim=D`。若共有 `L` 个 multiplier，编码器中有 `L` 个 stage，但只在前 `L-1` 个 stage 末尾下采样；解码器对称地上采样 `L-1` 次。

| 区段 | 操作 | 输出形状（通用表示） |
|---|---|---|
| 输入 | `x` | `[B,C,F,H,W]` |
| stem | `Conv3d(C→D, kernel=(1,7,7))` | `[B,D,F,H,W]` |
| stem temporal | PreNorm + temporal MHA + residual | `[B,D,F,H,W]` |
| encoder stage `i` | 2× conditioned ResNet block | `[B,D*mᵢ,F,H/2ⁱ⁻¹,W/2ⁱ⁻¹]` |
| encoder attention | slice-wise spatial linear attention + temporal MHA | 同上 |
| encoder downsample | `Conv3d(kernel=(1,4,4), stride=(1,2,2))` | `F` 不变，`H,W` 减半 |
| bottleneck | ResBlock → full spatial MHA → temporal MHA → ResBlock | `[B,D*mₗ,F,H/2ᴸ⁻¹,W/2ᴸ⁻¹]` |
| decoder stage `i` | 与 skip 拼接 → 2× conditioned ResNet block → 两类 attention | 通道恢复为上一层 |
| decoder upsample | `ConvTranspose3d(kernel=(1,4,4), stride=(1,2,2))` | `F` 不变，`H,W` 加倍 |
| final skip | 与 stem 保存的 `r` 拼接 | `[B,2D,F,H,W]` |
| head | unconditioned `ResnetBlock(2D→D)` + `Conv3d(D→C,1)` | `[B,C,F,H,W]` |

每个 encoder skip 是在两个 ResNet block 和两类 attention 之后、下采样之前保存的。

### 5.2 条件注入

时间步编码：

```text
t [B]
 -> SinusoidalPosEmb(D)
 -> Linear(D, 4D)
 -> GELU
 -> Linear(4D, 4D)
 -> time_embedding [B,4D]
```

直方图没有单独的投影、归一化或编码器，而是直接拼接：

```text
conditioning = concat(time_embedding [B,4D], histogram [B,K])
             = [B,4D+K]
```

每个 conditioned `ResnetBlock`使用 `SiLU + Linear(4D+K, 2*out_channels)` 产生 `(scale, shift)`，只调制该 ResNet block 的第一个 `Block`：

```text
h = GroupNorm(Conv3d(x))
h = h * (scale + 1) + shift
h = SiLU(h)
```

U-Net 同时创建了可学习 `null_cond_emb [1,K]`，为 classifier-free guidance 预留。但当前 `model.py` 训练和采样都直接调用 `denoiser.forward()`，`null_cond_prob` 保持默认值 0，也未调用 `forward_with_cond_scale()`，所以该 null embedding 在当前路径中既不会获得有效训练，也不用于 guidance。

### 5.3 ResNet 与注意力块

| 组件 | 实现 | 作用 |
|---|---|---|
| `Block` | `(1,3,3) Conv3d → GroupNorm(8) → 可选 FiLM → SiLU` | 切片内局部特征提取 |
| `ResnetBlock` | 2× `Block` + identity/`1×1×1 Conv3d` 残差支路 | 通道变换和条件注入 |
| `SpatialLinearAttention` | 将 `[B,C,F,H,W]` 变为 `[B*F,C,H,W]` 做线性注意力 | 每张切片内的全局空间建模 |
| temporal `Attention` | 将每个 `(h,w)` 位置的 `F` 个 token 做多头自注意力 | 跨切片信息交互 |
| bottleneck spatial `Attention` | 每个切片对 `H'*W'` token 做完整多头注意力 | 低分辨率全局空间建模 |
| `RelativePositionBias` | 32 buckets，`max_distance=32` | 给 temporal attention 添加切片相对位置偏置 |
| `RotaryEmbedding` | 默认 rotary dim 32 | 给 temporal query/key 注入位置信息 |
| `LayerNorm` | 沿通道维手写归一化，仅有可学习 `gamma` | attention 的 PreNorm |

默认 attention 为 8 头、每头 32 维，内部 hidden dimension 为 256；包装器未暴露这些参数的配置入口。

## 6. 函数与类对照

### 6.1 `model.py`

| 符号 | 职责 | 关键行为/副作用 |
|---|---|---|
| `require_torch()` | 返回已导入的 `torch` 模块 | 兼容接口；并不延迟 PyTorch 导入 |
| `load_official_unet()` | 动态导入 vendored `Unet3D` | 把 vendor root 插入全局 `sys.path[0]`，然后导入 `ddpm.diffusion` |
| `_extract()` | 根据 batch 中每个 `t` 提取扩散系数 | 输出 reshape 为 `[B,1,1,1,1]` 以便广播 |
| `LeFusionH.__new__()` | 模型工厂 | 每次调用都定义并返回一个局部 `_LeFusionH(nn.Module)` 实例 |
| `_LeFusionH.__init__()` | 创建去噪器和 DDPM buffers | 使用线性 beta schedule |
| `_LeFusionH.q_sample()` | DDPM 前向加噪 | 支持外部传入 noise 以便复现 |
| `_LeFusionH.forward()` | 训练入口 | 返回 mask 前景归一化噪声损失，不返回预测张量 |
| `_LeFusionH._p_mean_variance()` | 单步反向扩散参数 | 从噪声预测恢复 `x̂₀`，固定 clamp 到 `[-1,1]` |
| `_LeFusionH.sample_patch()` | mask 区域的完整反向采样 | 每步在 mask 外注入带噪背景，最后恢复精确背景 |
| `load_model_checkpoint()` | 构建模型、加载权重并切换 eval | 默认尝试 `ema_model`，不存在时静默回退到 `model` |

### 6.2 直接引用的其他符号

| 符号 | 文件 | 职责 |
|---|---|---|
| `masked_foreground_loss()` | `losses.py` | 验证 prediction/target/mask 的部分形状，扩展单通道 mask，计算每样本前景归一化 L1/L2 |
| `get_logger()` | `logger.py` | 获取 logger；若 root logger 无 handler，在导入期间安装 stderr INFO handler |
| `Unet3D` | vendored `ddpm/diffusion.py` | 条件噪声预测器 |
| `SinusoidalPosEmb` | 同上 | 扩散时间步编码 |
| `Block` / `ResnetBlock` | 同上 | 平面卷积、GroupNorm、SiLU、FiLM 与残差连接 |
| `SpatialLinearAttention` | 同上 | 逐切片空间线性注意力 |
| `Attention` | 同上 | 标准多头自注意力 |
| `RelativePositionBias` | 同上 | temporal attention 的分桶相对位置偏置 |
| `LayerNorm` / `PreNorm` / `Residual` | 同上 | attention 前归一化和残差包装 |
| `EinopsToAndFrom` | 同上 | 在空间/temporal attention 前后变换张量布局 |
| `Downsample` / `Upsample` | 同上 | 仅对 `H,W` 做 2 倍下/上采样 |

## 7. 问题与风险

下列优先级表示建议处理顺序。“确定”表示只根据已审查代码就能确认；“条件性”表示是否实际触发取决于未在审查范围内的配置或数据。

### P1：高优先级

#### 7.1 `image_size` 被当作基础通道数（确定语义冲突，影响为条件性）

`model.py` 调用：

```python
Unet3D(dim=int(config["image_size"]), ...)
```

而 `Unet3D` 用 `dim` 生成通道宽度 `D*mᵢ`、时间嵌入宽度 `4D` 和绝大多数卷积层；它不用 `dim` 校验 `H,W`。若 `image_size` 确实是补丁边长（例如 128 或 256），就会意外建立极宽的 U-Net，显著增加参数、显存和注意力计算量。

**建议：**把该配置项明确拆成 `base_dim` 与 `image_size/patch_size`，并用 checkpoint 中第一层卷积权重形状核对实际 `D`。

#### 7.2 checkpoint 反序列化可执行不可信代码（确定）

`torch.load(..., weights_only=False)` 允许完整 pickle 反序列化。加载网络下载、第三方或来源不可验证的 checkpoint 时，这可导致任意代码执行。

**建议：**仅允许可信 checkpoint；若 checkpoint 结构兼容，改为 `weights_only=True`，并对顶层 key 和 state dict 类型做白名单验证。

#### 7.3 mask 不是显式网络条件（确定）

训练时 `self.denoiser(x=noisy, time=timestep, cond=histogram)` 不接收 mask；mask 只决定哪些体素产生损失。采样时 mask 同样不传入 U-Net，而是通过 `torch.where` 限制生成区域。这意味着 U-Net 不能直接感知目标病灶边界或形状，只能从当前带噪图像、直方图和采样时的区域覆盖间接获得约束。

对严格的“按指定 mask 形状合成”任务，这是概念上的不足，可能表现为边界不连续、小病灶形状不稳定或 mask 内的生成内容与指定几何不一致。

**建议：**在确认 checkpoint 兼容策略后，考虑将 mask/距离变换作为额外输入通道或多尺度条件；至少要在模型说明中明确它是“区域约束采样”而非“mask-conditioned U-Net”。

#### 7.4 缺少核心形状/配置约束验证（确定）

当前代码依赖但未显式检查：

- `D` 必须适配 `SinusoidalPosEmb` 和 `GroupNorm(8)`。实用上 `D` 及每个 `D*mᵢ` 应为 8 的倍数；`D` 为奇数时，正弦编码输出长度为 `2*floor(D/2)`，与后续 `Linear(D,4D)` 不匹配。
- `H,W` 应同时能被 `2^(L-1)` 整除；否则奇数尺寸经下采样丢失一个体素后，解码时可能无法与 skip 对齐拼接。
- `histogram` 应为 `[B,K]`，且 batch 数、最后一维、device 和 dtype 都应与模型兼容。
- `timestep` 应为 `[B]` 的 long tensor，取值应在 `[0,T-1]`。
- `T` 应为正整数，并需让最后的 `ᾱ` 足够接近 0，才与从标准高斯噪声开始的采样假设匹配。

这些问题当前只会在深层卷积、GroupNorm、线性层、`torch.cat` 或 `gather` 中以低可读性异常暴露。

### P2：中优先级

#### 7.5 `LeFusionH` 不是真正的 `nn.Module` 类（确定）

`LeFusionH.__new__()` 在函数内部定义 `_LeFusionH(nn.Module)` 并返回它，因此：

- `isinstance(model, LeFusionH)` 为 false。
- 每次构建模型都会创建新的 Python 类对象。
- 局部类无法被标准 pickle 按全限定名稳定导入，直接序列化整个模型可失败。
- 类型检查、注册、文档生成和部分编译/分布式工具的行为更难预测。

**建议：**将 `_LeFusionH` 提升为模块顶层的正常 `class LeFusionH(nn.Module)`；仅在 `__init__` 中延迟加载 vendored U-Net。

#### 7.6 vendored 导入会污染全局模块解析（确定）

`load_official_unet()` 把 vendor root 永久插入 `sys.path[0]`，再以通用顶层名 `ddpm.diffusion` 导入。风险包括：

- 后续导入的同名包可被 vendor 版本意外遮蔽。
- 如果 `sys.modules` 中已经有另一个 `ddpm`，可能静默复用错误模块，而不是加载目标 vendor 文件。
- 导入 `ddpm.diffusion` 会执行整个大模块的顶层导入，即使当前只需要 `Unet3D`。

**建议：**将 vendor 源树作为明确包路径安装/导入，或使用唯一包名；至少在导入后校验 `Unet3D.__module__` 与模块 `__file__` 是否指向预期路径。

#### 7.7 直方图条件缺少归一化、编码和 dtype 对齐（确定实现，影响为条件性）

`histogram.to(image.device)` 只移动设备，不校验形状，也不转换到模型所需 dtype。条件被直接与时间嵌入拼接，没有 LayerNorm/MLP 编码。若不同 bin 量纲差异大、直方图未归一化、传入 `float64` 或整型，可能导致条件主导/失效、额外类型提升或线性层 dtype 错误。

**建议：**在模型入口校验 `[B,K]`，对齐到模型参数 dtype/device，并明确直方图归一化契约；若允许调整架构，使用独立 condition encoder。

#### 7.8 采样 mask 与训练 mask 契约不一致（确定）

`masked_foreground_loss()` 允许 `[B,1,F,H,W]` mask 自动扩展到 `C` 通道；`sample_patch()` 却要求 `background.shape == mask.shape`。这会使同一份单通道 mask 能用于训练、却不能直接用于多通道采样。

另外，`mask.to(bool)` 会把所有非零值（包括负值和很小的软权重）当作生成区域；训练损失却把 mask 作为浮点权重，两者语义不同。

**建议：**统一为明确的 binary mask 契约，训练和采样共用同一个形状/值域验证与通道扩展函数。

#### 7.9 所谓 RePaint 没有 jump/resampling（确定）

`sample_patch()` 只做一次从 `T-1` 到 0 的单调反向遍历，没有 RePaint 的反向/前向 jump 和多次 resampling。每个时间步的背景噪声也都独立重新采样，不是一条显式保持一致的前向噪声轨迹。

**建议：**若需要算法级 RePaint 对齐，实现明确的 jump schedule 并评估边界调和；否则将方法命名为 simplified background-injected/mask-clamped DDPM，避免夸大算法等价性。

#### 7.10 固定线性 beta 端点与任意 `T` 组合不一定合理（条件性）

采样固定从 `N(0,I)` 开始，但 beta 始终从 `1e-4` 线性到 `2e-2`。当 `T` 较小时，最终 `ᾱ_(T-1)` 可能仍显著大于 0，训练前向过程的末态就不接近纯高斯噪声，与采样起点存在分布错配。

**建议：**构建时计算并记录最终 `ᾱ`，对过大值报错/告警；让 schedule 与 `T` 联动设计，并保证训练和采样/checkpoint 使用同一 schedule。

#### 7.11 固定 `[-1,1]` clamp 依赖未显式声明的数据契约（条件性）

`_p_mean_variance()` 总是将预测的干净影像 clamp 到 `[-1,1]`。只有当训练和推理影像均严格使用该强度范围时才合理；否则会造成强度截断和训练/采样不一致。

**建议：**在模型配置中显式声明 `data_range`，加载 checkpoint 时一起校验，避免由外部数据管线默契维持。

#### 7.12 注意力的计算/显存伸缩性（确定架构，影响为条件性）

- temporal attention 在每个平面位置对 `F` 个 token 做标准自注意力，关键复杂度随 `H*W*F²` 增长；且 `F` 从不下采样。
- bottleneck spatial attention 对每张切片的 `H'*W'` token 做完整自注意力，关键复杂度随 `F*(H'W')²` 增长。
- stem 与每个 encoder/decoder stage 都有 temporal attention，高分辨率层可成为显存瓶颈。

**建议：**对实际 `F,H,W,D,M` 记录峰值显存和单步时延；如需扩展到更长 3D 体数据，考虑 windowed/chunked attention、深度方向采样或更明确的 3D 局部建模。

### P3：低优先级/可维护性

#### 7.13 EMA 缺失时静默回退（确定）

`use_ema=True` 但 checkpoint 中无 `ema_model` 或其值为空时，代码会静默使用 `checkpoint["model"]`，但日志仍只显示 `use_ema=True`。这会让实验记录误以为使用了 EMA。

**建议：**请求 EMA 时若缺少则显式报错，或至少记录 warning 和最终选中的 key。

#### 7.14 `Unet3D.forward(cond=None)` 的 fallback 不通用（确定，当前包装路径通常不触发）

vendored `Unet3D` 在 `cond is None` 时在 CPU 上创建固定 `[1,16]` 向量。它忽略 batch size 和实际 `cond_dim=K`，只在特定 `K=16` 及可广播场景下可能工作。`model.py` 总是传 histogram，所以这主要是核心 U-Net 公共接口的脆弱性。

#### 7.15 导入时日志副作用（确定）

`logger = get_logger(__name__)` 可在 root logger 尚无 handler 时立即安装 stderr handler 并把 root level 设为 INFO。导入一个模型模块因此会改变宿主应用全局日志行为。

#### 7.16 `require_torch()` 的延迟依赖语义已失效（确定）

`torch` 在模块顶层已经被 eager import，因此 `require_torch()` 只是返回已导入对象，无法把缺少 PyTorch 的错误延迟到模型构建时。

## 8. 建议的修复顺序

1. **先核对 checkpoint 和配置语义**：确认 `image_size` 实际数值及第一层权重通道数，拆分 `base_dim`/`patch_size`，固化线上 checkpoint 结构。
2. **收紧安全与加载契约**：不加载不可信 pickle，明确 EMA 选择，验证 state dict 和 DDPM buffer 尺寸。
3. **增加单一的模型输入验证层**：检查 `D/M/T`、`[B,C,F,H,W]`、`[B,K]`、mask 值域/通道、dtype/device 和 `H,W` 可整除性。
4. **将动态局部类改为顶层 `nn.Module`**，同时消除全局 `sys.path` 污染。
5. **明确任务条件设计**：决定 mask 是否应进入 U-Net；若不进入，将模型准确定义为直方图条件、mask-clamped 合成。
6. **最后再改变采样算法/架构**：根据边界质量实验决定是否需要真正 RePaint jumps、mask encoder、condition encoder 或更可扩展的 3D attention。这些改动会影响 checkpoint 兼容性，应在前四项稳定后进行。

## 9. 当前代码无法单独回答的信息

由于遵守本次审查边界，以下内容不应从 `model.py` 猜测：

- 当前实际 `D/C/K/T/M` 数值、精确参数量和显存占用。
- 训练影像是否真正归一化到 `[-1,1]`。
- histogram 的统计定义、归一化方式和临床意义。
- mask 是 binary、soft weight 还是多类标签，以及训练时是否存在空 mask。
- checkpoint 是由当前包装器还是其他实现生成，及其 EMA key 的真实命名。
- 当前有无测试覆盖边界尺寸、多通道 mask、混合精度、checkpoint 回载和可复现采样。

若后续允许扩大审查范围，应优先用实际 config 与 checkpoint tensor shape 回填第 3、5 节，然后才能给出精确层数、各层尺寸、参数量和实际风险是否已触发的结论。
