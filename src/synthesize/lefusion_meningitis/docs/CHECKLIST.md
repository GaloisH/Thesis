# Agent 项目清单


## 当前已实现

### 2026-08-11 病灶皮层放置（计划：`plans/2026-08-11-lesion-placing-plan.md`，状态：已执行）

**变更摘要**：首版将合成病灶位置选择改为 FastSurfer 皮层体素约束，替代原 `choose_candidate()` 的 position-prior 全脑采样。

**涉及文件**：
- 新增 `synthesis/cortical_placement.py`：`read_cortical_label_ids` / `load_cortical_mask` / `choose_cortical_candidate` / `placement_report`
- 新增 `tests/test_cortical_placement.py`（单元测试，15 项）与 `tests/verify_cortical_pipeline.py`（接受路径端到端验证）
- 修改 `synthesis/pipeline.py`：每病例进入病灶循环前调用一次 `load_cortical_mask`；分割缺失时报错并跳过该病例；`choose_candidate` → `choose_cortical_candidate`；接受病灶时把 `placement_report()` 写入每病灶 metadata；病例 metadata 写入 `fastsurfer_segmentation` 与 `fastsurfer_lut`
- 修改 `synthesis/__init__.py`：导出新增四个函数
- 修改 `config/lefusion_meningitis.yaml`（data 段）与 `config.py`（路径解析列表）：新增 `fastsurfer_subjects_dir`、`fastsurfer_segmentation_template`、`fastsurfer_lut`

**验证命令及结果**（计划验证项 1–5 全部通过）：
1. `read_cortical_label_ids()` 从仓库 LUT 读到 `1002`/`2002` 且不含白质 `2` —— 通过
2. `load_cortical_mask()` 对对齐 toy 分割正确识别 `ctx-*` 体素；对不同网格 toy 分割最近邻重采样后 shape 与参考影像一致 —— 通过
3. `choose_cortical_candidate()` 返回 ROI 满足 `np.all(cortical_mask[roi][donor_mask])`，并拒绝越界、非皮层、保护区重叠 —— 通过
4. 固定随机 seed 两次调用返回相同候选中心 —— 通过
5. 小规模 `synthesize` 运行：无 FastSurfer 分割时 28/28 病例以 `cortical_mask_unavailable` 跳过、`rejected_attempts=0`；为 case_000 注入对齐的合成皮层分割后，单病例接受路径 `cortical_fraction==1.0`、`cortical_voxels==lesion_voxels`、`qc.background_exact==true`、QC 无失败 —— 通过

```
python src/synthesize/tests/test_cortical_placement.py            # 15/15 passed
python src/synthesize/tests/verify_cortical_pipeline.py           # ACCEPT-PATH VERIFICATION PASSED
python -m lefusion_meningitis --set ... synthesize                # 28/28 跳过（skip 行为）
```

**遗留事项**：
- 真实 FastSurfer 分割尚未就绪（`datasets/fastsurfer_subjects` 不存在），真实病例的全量 `synthesize` 运行未做；接受路径用注入的合成皮层分割验证
- `placement.py` 中的 `choose_candidate` / `compute_mask` 仍保留但不再被调用（本计划只要求替换调用，未删除）
- 首版未做 DKT 区域分布、曲面/厚度、病灶掩膜裁剪；若皮层限制导致成功率偏低，先按计划分析 metadata 再决定是否进入第二阶段

## 近期计划

- （可选）第二阶段：按 DKT 分区/半球建立位置先验，或引入曲面、更宽松的“主要位于皮层”规则
