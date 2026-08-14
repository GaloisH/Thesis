# 决策 0001：合成病灶放置在 FastSurfer 皮层内（体素级约束）


## 背景

现有 `synthesize` 的病灶位置由 `placement.choose_candidate()` 依据 `position_prior.json` 的全脑中心采样决定，无法保证“病灶位于皮层”这一解剖约束。本仓库已有 FastSurfer 分割输出（`mri/aparc.DKTatlas+aseg.deep.mgz` 与 `FastSurfer_ColorLUT.tsv`），计划要求首版采用最小、可自动验证的体素级皮层约束，不引入曲面、皮层厚度或区域先验。

## 决策

1. 病灶中心从 FastSurfer 分割中名称以 `ctx-lh-`/`ctx-rh-` 开头的皮层体素内均匀抽取；要求供体病灶的全部体素落在皮层掩膜内（`cortical_fraction == 1.0`），并把 `placement_report()` 写入每条病灶 metadata。
2. 缺失分割文件或皮层掩膜为空的病例：报错并跳过该病例（`failed_cases` 记 `reason=cortical_mask_unavailable`），不猜测路径、不回退到全脑随机位置；放置尝试达到上限后记录失败。
3. 分割与参考影像 shape/affine 不一致时，用 nibabel `resample_from_to(order=0)` 最近邻重采样到参考影像网格。
4. `fastsurfer_segmentation_template` 保持相对路径，在 pipeline 中与 `fastsurfer_subjects_dir` 拼接；不加入 `config.py` 的绝对路径解析列表。`fastsurfer_subjects_dir` 与 `fastsurfer_lut` 仍走绝对路径解析。
5. 保留 `transform_donor_mask()`、采样与 patch QC；用 `choose_cortical_candidate()` 替代 `choose_candidate()` 的位置选择职责，已有病灶与已插入病灶仍通过 `current_label` 与 `protected_dilation` 避让。

## 原因

- “病灶在皮层上”需要可直接自动检查的定义；体素级皮层标签即可满足首版，且来自 FastSurfer 的既有输出。
- 若把模板也解析成绝对路径，`subjects_dir / template` 拼接时模板的绝对路径会遮蔽 subjects_dir 前缀（已实测复现该 bug），故模板保持相对路径拼接。
- 首版不做裁剪/形态修改/医学合理性判定，降低风险并保证可复现（固定 seed 时放置结果确定）。

## 影响

- `synthesize` 现在依赖每个目标病例存在对应的 FastSurfer 分割；缺失时该病例被跳过而非生成，汇总 `summary.json` 记录失败原因。
- 每病灶 metadata 新增 `placement`（`center`/`cortical_candidate_voxels`/`lesion_voxels`/`cortical_voxels`/`cortical_fraction`）；病例 metadata 新增 `fastsurfer_segmentation` 与 `fastsurfer_lut`。
- 输出影像、标签语义、模型与训练、其余 QC/输出格式均未改变。
- 遗留：真实 FastSurfer 分割未就绪，真实病例全量运行待数据到位后验证；首版成功率需通过 metadata 分析，再决定是否进入第二阶段的区域分布或曲面策略。
