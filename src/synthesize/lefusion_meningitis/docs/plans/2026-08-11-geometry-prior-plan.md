# 计划：将纯几何 QC 检查前置，避免重试时重复跑模型采样

- 日期：2026-08-11
- 状态：待实现
- 原则：最小改动；只前置「不依赖模型生成结果」的检查，不改生成、放置与 QC 语义。

## 目标

当前 `synthesize()` 的每个病灶重试流程为：选供体 → 变换 mask → 选皮层位置 →
跑 diffusion 采样（约 300 步，最贵）→ `qc_patch` 生成后检查。任何 QC 失败都会
丢弃本次结果并重跑一次模型采样。

`qc_patch` 中的 `mask_too_small` 与 `mask_touches_patch_edge` 是纯几何检查，只由
`transform_donor_mask` 的输出决定。由于放置时 ROI 大小恒等于供体 mask 大小，这两
个条件与放置位置无关，可以在跑模型之前就判定。目标是把它们前置，让这两类廉价失败
不再消耗模型采样；`qc_patch` 保留原检查作为兜底。

## 核心函数与职责

### `synthesis/placement.py`：`reject_donor_geometry(mask, *, min_voxels) -> str | None`

对变换后的供体 mask 做纯几何先验判定，返回失败原因字符串或 `None`：

- mask 为空 → `"mask_empty"`；
- `mask.sum() < min_voxels` → `"mask_too_small"`；
- mask 触碰自身数组边缘（等价于触碰 ROI 边缘）→ `"mask_touches_patch_edge"`；
- 全部通过 → `None`。

返回原因字符串（而非布尔）便于日志与 metadata 统计失败分布。

### `synthesis/pipeline.py`：前置调用

在 `transform_donor_mask` 之后、`robust_normalize` / 位置选择 / 模型采样之前调用
`reject_donor_geometry`；失败时计入 `case_rejections` / `rejected_attempts` 并
`continue` 下一次尝试，与现有 RuntimeError / QC 失败的处理一致。阈值取
`data.min_component_voxels`（默认 8），与现有 QC 硬编码值一致。

## 验收标准

1. 单元测试：对小于阈值的 mask、触碰边缘的 mask、正常 mask，`reject_donor_geometry`
   分别返回对应原因 / `None`；
2. `mask_touches_patch_edge` 判定与位置无关：任意平移 mask 前后结论不变；
3. 前置失败时不会调用 `robust_normalize`、位置选择或 `model.sample_patch`；
4. 原 `qc_patch` 中的 `mask_too_small` / `mask_touches_patch_edge` 保留为兜底，
   前置通过后不再触发；
5. 固定 seed 时，最终合成的病例 / 病灶数量与改动前一致（先验只拒绝原本会在 QC
   阶段被拒绝的候选）。
