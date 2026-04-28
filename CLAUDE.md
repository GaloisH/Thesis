# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical image analysis pipeline for brain tumor (meningioma) segmentation and registration. The project processes multi-modal MRI scans (T1, T1ce, T2, FLAIR) and performs:
1. **Segmentation**: Ensemble of nnUNet and SwinUNETR models
2. **Registration**: ANTs SyN registration to MNI152NLin2009cAsym space
3. **Statistics**: Lesion frequency/probability map construction with brain region analysis

## Environment Setup

Set nnUNet environment variables before running any nnUNet commands:
```bash
export nnUNet_raw=/path/to/thesis/datasets/nnUNet_raw
export nnUNet_preprocessed=/path/to/thesis/datasets/nnUNet_preprocessed
export nnUNet_results=/path/to/thesis/datasets/nnUNet_results
```

The Python scripts in `src/` auto-bootstrap these paths when run directly.

## Common Commands

### Segmentation

**Run nnUNet inference with softmax output:**
```bash
cd src
python nnunet_softmax_predict.py \
  --input_dir ../datasets/nnUNet_raw/Dataset101_Meningioma/imagesTs \
  --output_dir ../outputs/nnunet_softmax \
  --model_dir ../datasets/nnUNet_results/Dataset101_Meningioma/nnUNetTrainer__nnUNetPlans__2d \
  --folds 0 \
  --device cuda
```

**Run ensemble prediction (nnUNet + SwinUNETR):**
```bash
cd src
python ensemble_predict.py \
  --input_dir ../datasets/nnUNet_raw/Dataset101_Meningioma/imagesTs \
  --output_dir ../outputs/ensemble \
  --swin_checkpoint /path/to/best_model.pth \
  --w_nnunet 0.6 --w_swin 0.4 \
  --device cuda
```

**Train SwinUNETR:**
Update paths in `src/swin_train.py` __main__ section, then:
```bash
cd src
python swin_train.py
```

### Registration

**Run registration pipeline (from registration_simple):**
```bash
cd src/registration_simple
python registration.py \
  --max-cases 10 \
  --output-dir ../../datasetsregistration \
  --atlas-path ./atlas_ready/AAL3v2.nii.gz \
  --atlas-labels ./atlas_ready/AAL3v2.json
```

**Generate visualizations:**
```bash
cd src/registration_simple
python visualization.py \
  --output-dir ../../datasetsregistration \
  --viz-dir ../../datasetsregistration/visualizations
```

## Data Structure

### Input Data (nnUNet format)
```
datasets/nnUNet_raw/Dataset101_Meningioma/
├── dataset.json           # Channel names and label definitions
├── imagesTr/              # Training: {case_id}_{channel:04d}.nii.gz
│   ├── case_001_0000.nii.gz  # T1
│   ├── case_001_0001.nii.gz  # T1ce
│   ├── case_001_0002.nii.gz  # T2
│   └── case_001_0003.nii.gz  # FLAIR
├── labelsTr/              # Labels: {case_id}.nii.gz (0=bg, 1=necrotic, 2=edema, 3=enhancing)
└── imagesTs/              # Test images (same format)
```

### Output Structure
```
outputs/ensemble/
├── ensemble_seg/          # Final segmentation masks
├── ensemble_softmax/      # Fused softmax probabilities (.npz)
└── swin_softmax/          # SwinUNETR probabilities (.npz)

datasetsregistration/
├── lesion_probability_map.nii.gz
├── lesion_frequency_map.nii.gz
├── region_probability_by_atlas.csv
├── checkpoint.json        # Progress for resumable registration
└── registered/            # Per-case MNI-space outputs
```

## Architecture

### Segmentation (`src/`)
- `nnunet_softmax_predict.py`: nnUNet inference wrapper, extracts softmax probabilities
- `ensemble_predict.py`: Weighted fusion of nnUNet and SwinUNETR softmax outputs
- `swin_train.py`: SwinUNETR training with MONAI transforms, wandb logging
- `plan2transform.py`: Converts nnUNetPlans.json to MONAI transform pipelines

### Registration (`src/registration_simple/`)
- `registration_core.py`: Core ANTs SyN functions, parallel processing, checkpoint management
- `registration.py`: CLI entry point
- `visualization.py`: Orthogonal slices, overlays, histograms, QC plots

### Key Registration Parameters
- Transform: ANTs SyN (Symmetric Normalization)
- Template: MNI152NLin2009cAsym (2mm, brain-extracted) via TemplateFlow
- Atlas: AAL3v2 (170 brain regions) in `atlas_ready/`
- Parallel: ProcessPoolExecutor with memory-aware worker allocation (~4GB per worker)

## Label Definitions

Segmentation labels (from dataset.json):
- 0: background
- 1: necrotic
- 2: edema
- 3: enhancing_tumor

The `ConvertToBratsRegionsd` transform converts integer labels to 4-channel one-hot format for SwinUNETR training.

## Dependencies

Key packages: `torch`, `nnunetv2`, `monai`, `ants` (ANTsPy), `nibabel`, `templateflow`, `wandb`