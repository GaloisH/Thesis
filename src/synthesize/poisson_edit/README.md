# 3D 泊松病灶融合实验

该实验从 `lesion_ldm/preprocess/0721` 选择一组 `32³` 病灶图像和 mask，
将其插入到 Dataset002 的另一病例中，并比较直接复制与真正的三维泊松融合。

## 实现

- SimpleITK：NIfTI 读写、空间信息继承和 Otsu 前景提取；
- SciPy：三维形态学、稀疏拉普拉斯矩阵及 `spsolve`；
- NumPy：体素计算和局部 median/IQR 强度匹配；
- Matplotlib：无界面的二维对比图。

对 mask 内每个体素使用 6 邻域离散泊松方程，mask 外的目标图像作为
Dirichlet 边界。项目没有使用 OpenCV `seamlessClone`，因为它只能逐张处理
二维图像，无法保证相邻切片连续。

## 运行

在项目根目录执行：

```powershell
python -m src.synthesize.poisson_edit.run_experiment
```

默认固定 `seed=42`，使用通道 2，并输出到：

```text
outputs/poisson_edit/seed_42/
├── poisson_blended.nii.gz
├── copy_paste.nii.gz
├── inserted_mask.nii.gz
├── comparison.png
└── metadata.json
```

可通过以下参数复现或指定数据：

```powershell
python -m src.synthesize.poisson_edit.run_experiment `
  --seed 42 `
  --channel 2 `
  --source-mask src/synthesize/lesion_ldm/preprocess/0721/case_000_lesion009_mask.nii.gz `
  --target-image datasets/nnUNet_raw/Dataset002_Meningitis/imagesTr/case_001_0002.nii.gz `
  --output-dir outputs/poisson_edit/custom
```

`metadata.json` 会记录 donor/target、插入坐标、强度匹配统计、泊松方程残差
以及两种方法在 mask 边界处的平均强度跳变。
