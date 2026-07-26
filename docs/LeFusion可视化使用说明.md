# LeFusion-H 指定 Mask 合成与可视化

## 单病例

输入目标单通道影像和与其 shape、affine 完全对齐的三维 NIfTI mask：

```powershell
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis `
  visualize `
  --image datasets/nnUNet_raw/Dataset002_Meningitis/imagesTr/case_000_0000.nii.gz `
  --mask path/to/lesion_mask.nii.gz `
  --output-dir outputs/lefusion_meningitis/visualizations/case_000 `
  --seed 42
```

mask 指定病灶形状与位置。程序自动提取 `32×32×32` patch，从训练直方图库采样
纹理条件，使用 `best.pt` 的 EMA 权重生成病灶，并只替换 mask 内体素。

每例输出：

```text
case_xxx/
├── original.nii.gz
├── generated_patch.nii.gz
├── synthetic.nii.gz
├── inserted_mask.nii.gz
├── metadata.json
└── figures/
    ├── 01_mask_orthogonal.png
    ├── 02_generated_lesion.png
    ├── 03_before_after_full.png
    ├── 04_before_after_zoom.png
    ├── 05_multislice_axial.png
    ├── 06_intensity_qc.png
    └── comparison.png
```

## 默认批量运行

不提供 `--image` 和 `--mask` 时，程序自动读取
`datasets/lefusion_meningitis/prepared/split.json`，合并其中的训练集与验证集病例，并从
配置的 nnUNet 数据集 `imagesTr` 和 `labelsTr` 中读取影像与标签。由于一个完整标签
可能包含许多相距较远、无法共同放入 `32³` patch 的病灶，程序会根据 prepared
manifest 为每个病例自动选择体素数最大的有效病灶组件作为生成 mask：

```powershell
D:\python_code\miniconda\python.exe -m src.synthesize.lefusion_meningitis `
  visualize `
  --output-dir outputs/lefusion_meningitis/visualizations
```

不需要准备 CSV 文件。批量目录会额外生成 `index.csv` 和可直接打开的 `index.html`
图片索引。

若 mask 为空、几何不一致、无法带边距放入模型 patch，或过于靠近影像边缘，命令会在
模型推理前给出明确错误。默认情况下 QC 不通过的样本仍会保存，以便通过诊断图定位问题。
