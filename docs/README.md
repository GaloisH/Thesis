# 病灶合成项目文档

本目录集中保存三维病灶扩散合成相关的论文解读、文献调研、实验设计与代码规划。

## 建议阅读顺序

1. [3D病灶扩散文献调研与技术路线](3D病灶扩散文献调研与技术路线.md)
   - 了解主要病灶合成路线、代表论文和当前技术选择。

2. [LeFusion论文方法详解](LeFusion论文方法详解.md)
   - 深入理解病灶聚焦扩散、背景保持、纹理控制和 DiffMask。

3. [数据合成PLAN](数据合成PLAN.md)
   - 查看 `32³` patch、AutoencoderKL、无条件 latent diffusion、病灶融合和实验矩阵的完整方案。

4. [代码规划大纲](代码规划大纲.md)
   - 查看未来代码目录、模块职责、数据结构、CLI、checkpoint 和测试边界。

## 文档关系

```text
文献调研与技术路线
        │
        ├── LeFusion方法详解（重点参考方案）
        │
        ▼
无条件3D LDM实验方案
        │
        ▼
代码规划大纲
```

## 当前技术路线

当前计划使用：

- 单序列脑 MR；
- 每例裁剪若干 `32×32×32` 病灶 patch；
- 局部标准化病灶残差与 mask 双通道表示；
- MONAI `AutoencoderKL`；
- 无 condition 的 latent diffusion；
- 联合生成病灶纹理和 mask；
- 根据目标局部强度反归一化；
- 距离变换软融合；
- 通过 nnU-Net 下游分割实验评价合成数据。

## 当前代码状态

目前只建立代码目录脚手架和文档，没有 Python、YAML、checkpoint 或可执行实现。代码结构位于：

```text
src/synthesize/lesion_ldm/
```

配置和测试的预留目录为：

```text
config/lesion_ldm/
tests/lesion_ldm/
```

后续实现应遵循[《代码规划大纲》](代码规划大纲.md)，并保持原始数据只读。

