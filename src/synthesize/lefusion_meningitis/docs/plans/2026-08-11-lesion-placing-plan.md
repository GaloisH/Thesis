# 计划：将合成病灶放置在 FastSurfer 皮层分割内（最小实现）

- 日期：2026-08-11
- 状态：待实现
- 原则：先实现一个小而可验证的体素级约束；不引入曲面、皮层厚度、分区先验或额外医学规则。

## 已确认的 FastSurfer 接口与标签

已查看本仓库 `FastSurfer/` 的实现，首版以其实际默认输出为输入：

- `FastSurfer/run_fastsurfer.sh` 的 `--asegdkt_segfile` 默认输出为
  `$SUBJECTS_DIR/$SID/mri/aparc.DKTatlas+aseg.deep.mgz`；
- `FastSurfer/FastSurferCNN/run_prediction.py` 同样将该文件作为 `pred_name`，并以 `int16`
  保存分割；
- 默认 LUT 是
  `FastSurfer/FastSurferCNN/config/FastSurfer_ColorLUT.tsv`，其格式为
  `ID, LabelName, R, G, B, A`（TSV）；
- LUT 中 `LabelName` 以 `ctx-lh-` 或 `ctx-rh-` 开头的 ID 是左右大脑皮层分区标签，例如
  `1002 ctx-lh-caudalanteriorcingulate`、`2002 ctx-rh-caudalanteriorcingulate`；
- FastSurfer 的 `data_loader.data_utils.read_classes_from_lut()` 已实现该 TSV 的读取。本项目只
  复用相同的文件格式和命名规则，不把 FastSurfer 训练/推理代码引入 LeFusion 运行时依赖。

## 目标与非目标

目标是在现有 `synthesize` 流程中，从 FastSurfer 标记为皮层的体素中选择病灶中心来放置病灶。这样“病灶在皮层上”有一个直接、可自动检查的
定义。

首版不做以下事项：

- 不使用 `recon-surf` 曲面、法向、皮层厚度或 pial/white 文件；
- 不按 DKT 分区、半球或真实病灶统计建立位置先验；
- 不对病灶掩膜进行裁剪、形态修改或医学合理性判定；
- 不修改训练数据、训练模型、输出标签语义或现有影像 QC。
- Fastsurfer 不进行表面重建，只进行分割获取体素级标签

若皮层区域不足以容纳当前供体病灶，则该次尝试失败并继续重试；达到上限后记录失败，而不是
退回到全脑随机位置。

## 核心函数与职责

新建 `src/synthesize/lefusion_meningitis/synthesis/cortical_placement.py`，仅包含下列函数。

### `read_cortical_label_ids(lut_path: Path) -> np.ndarray`

读取 FastSurfer TSV LUT，返回所有名称以 `ctx-lh-` 或 `ctx-rh-` 开头的整数 ID。

- 不使用 `1000–2999` 的数值范围猜测标签；
- 找不到文件、缺少 `ID`/`LabelName` 列或未解析到任何皮层标签时抛出明确异常；
- 返回的 ID 排序且去重，便于日志和单元测试复现。

### `load_cortical_mask(segmentation_path: Path, image_reference, lut_path: Path) -> np.ndarray`

读取 `aparc.DKTatlas+aseg.deep.mgz`，并生成与目标影像数组同 shape 的布尔型皮层掩膜。

1. 使用 nibabel 读取分割和目标影像；
2. 若两者 shape 与 affine 一致，直接读取分割数组；否则以最近邻插值将分割重采样到
   `image_reference` 网格；
3. 调用 `read_cortical_label_ids(lut_path)`，以 `np.isin(segmentation, cortical_ids)` 生成掩膜；
4. 掩膜为空时抛出异常。函数不读取或写入任何患者影像副本。

### `choose_cortical_candidate(cortical_mask, existing_label, donor_mask, rng, *, protected_dilation, max_attempts) -> (center, roi)`

替代当前 `placement.choose_candidate()` 的位置选择职责。

1. 从 `cortical_mask` 为真的体素中均匀抽取一个候选中心；
2. 计算与 `donor_mask` 同大小的 ROI；越界则拒绝；
3. 对 `existing_label > 0` 进行现有配置指定的膨胀，病灶与该保护区相交则拒绝；
4. 用固定 seed 时保持结果确定，并在异常信息中统计越界、非皮层和保护区重叠次数。

### `placement_report(cortical_mask, roi, donor_mask) -> dict`

返回可写入 metadata 的最小验证信息：`center`、皮层候选体素数、病灶体素数、
`cortical_voxels` 和 `cortical_fraction`。首版成功候选的 `cortical_fraction` 必须为 `1.0`。

## 对现有流程的最小改动

1. 在配置 `data` 段新增：

   ```yaml
   fastsurfer_subjects_dir: datasets/fastsurfer_subjects
   fastsurfer_segmentation_template: "{case_id}/mri/aparc.DKTatlas+aseg.deep.mgz"
   fastsurfer_lut: FastSurfer/FastSurferCNN/config/FastSurfer_ColorLUT.tsv
   ```

   `case_id` 与 nnUNet 病例 ID 一一对应；找不到分割文件的病例应报错并跳过，不猜测路径。

2. 在 `synthesis/pipeline.py` 每个目标病例进入病灶循环前调用
   `load_cortical_mask()` 一次，并在该病例内复用结果。

3. 保留 `transform_donor_mask()`、生成采样和现有 patch QC；将原有
   `choose_candidate()` 调用替换为 `choose_cortical_candidate()`。已有病灶和已插入病灶仍通过
   `current_label` 与 `protected_dilation` 避让。

4. 接受病灶时调用 `placement_report()`，将结果写入现有每病灶 metadata；病例 metadata 另写入
   FastSurfer 分割路径和 LUT 路径。其余输出格式保持不变。

5. 首版不改动 `prepare`、`position_prior.json` 或配置中的病灶尺度；若皮层限制导致成功率偏低，
   先通过 metadata 分析原因，再决定是否单独设计下一阶段的尺度或曲面策略。

## 验证步骤与验收标准

使用合成的微型 NumPy/NIfTI 体积编写单元测试，不使用原始医学数据：

1. `read_cortical_label_ids()` 能从仓库 LUT 读到 `1002` 与 `2002`，且不包含白质标签 `2`；
2. `load_cortical_mask()` 对已对齐的 toy 分割正确识别 `ctx-*` 体素；对不同网格的 toy 分割
   使用最近邻重采样后 shape 与参考影像一致；
3. `choose_cortical_candidate()` 返回的 ROI 满足
   `np.all(cortical_mask[roi][donor_mask])`，并拒绝越界、非皮层和保护区重叠的位置；
4. 固定随机 seed，两次函数调用返回相同候选中心；
5. 对一次小规模 `synthesize` 运行，逐项复核每条已接受病灶 metadata：
   `cortical_fraction == 1.0`、病灶与保护区不重叠，且掩膜外背景保持现有的逐体素不变检查。

通过以上检查后，再决定是否需要第二阶段的区域分布、皮层表面或更宽松的“主要位于皮层”规则。
