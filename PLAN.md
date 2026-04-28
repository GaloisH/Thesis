# 脑部MR图像病灶分析流程规划：分割-配准-统计-分析

## Context

**问题背景**: 需要开发一个完整的医学图像分析pipeline，用于脑部MRI病灶（脑肿瘤）的综合分析。目前项目已有分割和配准的基础实现，但缺乏统一的流程管理、完整的统计分析功能和自动化报告生成。

**应用场景**:
- 单患者分析：诊断报告、病灶定位、体积量化
- 群体研究：统计检验、组间比较、病灶模式聚类

**现有实现**:
- 分割：nnUNet + SwinUNETR ensemble (已实现)
- 配准：ANTs SyN → MNI152NLin2009cAsym (已实现)
- 统计：区域概率统计 (部分实现)
- 分析：缺失

**用户需求**:
- 报告格式：Markdown（简洁易读）
- 临床数据：暂无（后续可扩展）
- 优先级：流程整合优先（先让现有模块能完整运行）

---

## 实现方案

### 目录结构

```
thesis/src/
├── pipeline/                       # NEW: 流程管理
│   ├── config.py                   # YAML配置管理
│   ├── orchestrator.py             # 主流程协调器
│   ├── cli.py                      # 统一CLI入口
│   └── progress_tracker.py         # 进度追踪

├── segmentation/                   # NEW: 分割模块封装
│   ├── inference_engine.py         # 统一推理接口
│   ├── nnunet_wrapper.py           # nnUNet封装
│   ├── swin_wrapper.py             # SwinUNETR封装
│   ├── ensemble_fusion.py          # 集成融合策略
│   └── quality_control.py          # QC指标

├── registration_/                  # EXISTING: 增强
│   ├── registration_core.py        # 添加softmax变换支持
│   ├── multi_template.py           # 多模板配准
│   └── transform_cache.py          # 变换缓存

├── statistics/                     # NEW: 统计模块
│   ├── volumetric_stats.py         # 体积统计
│   ├── spatial_stats.py            # 空间分布统计
│   ├── population_stats.py         # 群体统计
│   └ statistical_tests.py         # 统计检验

├── analysis/                       # NEW: 分析模块
│   ├── lesion_burden.py            # 病灶负荷量化
│   ├── pattern_analysis.py         # 病灶模式分析
│   ├── group_comparison.py         # 组间比较
│   └── report_generator.py         # 报告生成

└── utils/                          # NEW: 公共工具
    ├── file_io.py
    ├── image_processing.py
    └── validation.py
```

---

## 核心模块设计

### 1. Pipeline Orchestration

**关键文件**: `pipeline/orchestrator.py`

工作流程:
1. SegmentationStep → 执行分割推理
2. QCStep → 分割质量验证
3. RegistrationStep → 配准到MNI空间
4. StatisticsStep → 计算统计数据
5. AnalysisStep → 分析并生成报告

**CLI命令设计**:
```bash
thesis-pipeline run --config config.yaml              # 完整流程
thesis-pipeline segment --input <dir> --output <dir>  # 仅分割
thesis-pipeline register --input <dir>                # 仅配准
thesis-pipeline stats --input <dir> --atlas AAL3v2    # 仅统计
thesis-pipeline analyze --report markdown             # 仅分析
```

### 2. Segmentation Enhancement

**关键文件**: `segmentation/inference_engine.py`

统一接口:
- `predict_single(image_paths)` → 单例推理
- `predict_batch(input_dir, output_dir)` → 批量推理
- `get_softmax_output()` → 概率图
- `get_quality_metrics()` → QC指标

复用现有代码:
- `nnunet_softmax_predict.py` → `nnunet_wrapper.py`
- `ensemble_predict.py` → `swin_wrapper.py` + `ensemble_fusion.py`

QC指标:
- DiceScoreQC, VolumeConsistencyQC, ConfidenceScoreQC

### 3. Statistics Module (重点开发)

**volumetric_stats.py** - 体积统计:
```python
compute_voxel_volume(seg, spacing, class_label) → mm³
compute_region_volumes(seg, atlas, labels) → 每区域体积
compute_volume_ratio(seg, brain_mask) → 病灶占比
```

**spatial_stats.py** - 空间统计:
```python
compute_centroid(seg, affine) → 世界坐标
compute_asymmetry_index(seg, midline_axis=0) → 左右不对称性
compute_lesion_spread(seg, centroid) → 病灶离散度
```

**population_stats.py** - 群体统计:
```python
compute_lesion_frequency_map(registered_masks) → 频率图
compute_region_lesion_frequency(masks, atlas) → 每区域病灶频率
```

### 4. Analysis Module (重点开发)

**lesion_burden.py** - 病灶负荷:
- 总病灶体积 (TLV)
- 病灶体积比 (LVR = TLV / brain_volume)
- 病灶数量和密度

**pattern_analysis.py** - 模式分析:
- 病灶热点区域识别
- 空间模式聚类
- 模式相似性分析

**report_generator.py** - 报告生成:
- Markdown报告 (简洁易读，支持图片链接)
- JSON报告 (结构化数据)

```python
def generate_markdown_report(results: Dict, output_path: str) -> None:
    """
    生成Markdown格式的综合分析报告
    - 患者基本信息
    - 分割结果摘要
    - 体积统计表
    - 空间位置图
    - 区域病灶分布
    """
```

---

## 数据流

```
输入数据 (4模态MRI + 标签)
       ↓
[分割模块] → ensemble_seg/*.nii.gz + ensemble_softmax/*.npz + qc_report.csv
       ↓
[配准模块] → registered/*_mni.nii.gz + transforms/*.mat
       ↓
[统计模块] → volumetric_stats.csv + spatial_stats.csv + lesion_probability_map.nii.gz
       ↓
[分析模块] → lesion_burden.csv + hotspot_regions.csv + report.md
       ↓
可视化输出 (正交切片、叠加图、直方图、条形图)
```

---

## 输出格式

### volumetric_stats.csv
```
subject_id,class_label,class_name,volume_mm3,volume_ratio,brain_volume_mm3
case_001,1,necrotic,1234.56,0.012,102400.00
case_001,2,edema,5678.90,0.055,102400.00
```

### spatial_stats.csv
```
subject_id,class_label,centroid_x,centroid_y,centroid_z,asymmetry_index
case_001,1,45.2,-12.3,78.9,0.15
```

### region_probability_by_atlas.csv (增强)
```
region_id,region_name,mean_probability,max_probability,lesion_frequency
1,Precentral_L,0.045,0.89,15
```

---

## 配置管理 (YAML)

```yaml
pipeline:
  input:
    data_dir: "datasets/nnUNet_raw/Dataset101_Meningioma"
  output:
    root_dir: "outputs/pipeline_results"

segmentation:
  ensemble:
    weights: {nnunet: 0.6, swin: 0.4}
  quality_control:
    min_dice_threshold: 0.85

registration:
  template: "MNI152NLin2009cAsym"
  atlas: "AAL3v2"
  n_jobs: -1

statistics:
  volumetric: {enabled: true}
  spatial: {enabled: true}
  population: {enabled: true}

analysis:
  report:
    format: "markdown"  # 简洁易读
    sections: [summary, volumetric, spatial, regional]
```

---

## 实现顺序 (优先流程整合)

### Phase 1: 流程整合 (优先 - 1周)
1. 创建目录结构
2. `pipeline/config.py` - 配置管理
3. `pipeline/orchestrator.py` - 流程协调器（整合现有模块）
4. `pipeline/cli.py` - 统一CLI入口
5. 端到端运行测试（使用现有分割和配准代码）

### Phase 2: 统计模块开发 (1周)
1. `statistics/volumetric_stats.py` - 体积统计
2. `statistics/spatial_stats.py` - 空间分布统计
3. `statistics/population_stats.py` - 群体统计（基于现有registration_core）

### Phase 3: 分析模块开发 (1周)
1. `analysis/lesion_burden.py` - 病灶负荷量化
2. `analysis/pattern_analysis.py` - 病灶模式分析
3. `analysis/report_generator.py` - Markdown报告生成

### Phase 4: 分割模块封装优化 (1周)
1. `segmentation/inference_engine.py` - 统一接口
2. 重构现有代码为wrapper类（可选，保持向后兼容）
3. `segmentation/quality_control.py` - QC指标

### Phase 5: 测试与文档 (0.5周)
1. 单元测试
2. 集成测试
3. 使用文档（README + CLI帮助）

### Phase 6: 功能增强 (后续)
1. 临床数据关联分析模块（预留扩展）
2. 多模板配准支持
3. 高级可视化（3D渲染等）

---

## 关键文件路径 (需创建/修改)

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/pipeline/config.py` | NEW | 配置管理核心 |
| `src/pipeline/orchestrator.py` | NEW | 流程协调器 |
| `src/pipeline/cli.py` | NEW | CLI入口 |
| `src/segmentation/inference_engine.py` | NEW | 统一推理接口 |
| `src/segmentation/nnunet_wrapper.py` | NEW | 重构自现有代码 |
| `src/segmentation/swin_wrapper.py` | NEW | 重构自现有代码 |
| `src/statistics/volumetric_stats.py` | NEW | 体积统计 |
| `src/statistics/spatial_stats.py` | NEW | 空间统计 |
| `src/analysis/lesion_burden.py` | NEW | 病灶负荷 |
| `src/analysis/report_generator.py` | NEW | 报告生成 |
| `src/registration_/registration_core.py` | MODIFY | 添加softmax变换 |
| `src/ensemble_predict.py` | MODIFY → REFACTOR | 拆分到模块 |

---

## 测试验证

1. **单元测试**: `tests/` 目录，每个模块独立测试
2. **集成测试**: 使用小样本数据验证端到端流程
3. **性能测试**: 验证并行处理效率
4. **输出验证**: 检查输出文件格式和数据完整性

**测试命令**:
```bash
pytest tests/ -v --cov=src/
thesis-pipeline run --config test_config.yaml --max-cases 5
```

---

## 依赖补充

需添加到 `requirements.txt`:
- `pyyaml` (配置)
- `click` (CLI增强)
- `scipy` (统计)
- `statsmodels` (统计检验)
- `rich` (进度显示)

---

## Markdown报告示例

```markdown
# 脑部病灶分析报告

## 患者: case_001

### 分割结果摘要
| 区域 | 体积(mm³) | 占比 |
|------|----------|------|
| 坏死区 | 1,234.56 | 1.2% |
| 水肿区 | 5,678.90 | 5.5% |
| 强化区 | 2,345.67 | 2.3% |

### 空间位置
- 中心坐标: (45.2, -12.3, 78.9) mm
- 左右不对称性: 0.15

### 病灶分布热点
![概率图叠加](./visualizations/probability_overlay.png)

| 脑区 | 平均概率 | 最大概率 |
|------|----------|----------|
| 额叶_L | 0.045 | 0.89 |
| 颞叶_R | 0.032 | 0.76 |
```

---

## Phase 1 详细任务清单 (流程整合)

1. **创建 `src/pipeline/` 目录结构**
   - `__init__.py`, `config.py`, `orchestrator.py`, `cli.py`

2. **实现 `config.py`**:
   - `PipelineConfig` 类，从YAML加载配置
   - 默认配置模板
   - 配置验证

3. **实现 `orchestrator.py`**:
   - `PipelineOrchestrator` 主类
   - 步骤定义：`run_segmentation()`, `run_registration()`, `run_statistics()`
   - 进度追踪和错误处理
   - **整合现有代码**：
     - 调用 `ensemble_predict.py` 主函数
     - 调用 `registration.py` 主函数

4. **实现 `cli.py`**:
   - `thesis-pipeline run --config <yaml>`
   - `thesis-pipeline segment --input <dir> --output <dir>`
   - `thesis-pipeline register --input <dir>`
   - 使用 `click` 库实现命令行界面

5. **创建默认配置文件 `config/default.yaml`**

6. **端到端测试**:
   - 使用 `--max-cases 5` 测试完整流程
   - 验证各阶段输出文件完整性