#!/usr/bin/env bash
# train.sh — 启动网络训练
# 用法: bash train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${ROOT_DIR}/datasets"

export nnUNet_raw="${OUTPUT_DIR}/nnUNet_raw"
export nnUNet_preprocessed="${OUTPUT_DIR}/nnUNet_preprocessed"
export nnUNet_results="${OUTPUT_DIR}/nnUNet_results"

export nnUNet_wandb_enabled=1                      # 启用 wandb 记录 (填 1 或 true)
export nnUNet_wandb_project="nnUNet_Meningioma"    # 在 wandb 网页端显示的项目名称 (可自由修改)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

export WANDB_NAME="Task101_3d_fullres_fold0_${TIMESTAMP}"

TASK_ID="101"

# 前置检查：预处理产物是否存在
PREPROCESS_DIR="${nnUNet_preprocessed}/Dataset${TASK_ID}_Meningioma"
if [[ ! -d "$PREPROCESS_DIR" ]] || [[ -z "$(ls -A "$PREPROCESS_DIR" 2>/dev/null)" ]]; then
    echo "❌ 未检测到预处理数据：${PREPROCESS_DIR}"
    echo "   请首先运行: bash preprocess.sh"
    exit 1
fi

echo "✅ 环境变量已设置:"
echo "   nnUNet_raw          = ${nnUNet_raw}"
echo "   nnUNet_preprocessed = ${nnUNet_preprocessed}"
echo "   nnUNet_results      = ${nnUNet_results}"

echo ""
echo "=============================================="
echo "启动网络训练 (Training)"
echo "=============================================="
echo ">> 执行命令: nnUNetv2_train ${TASK_ID} 3d_fullres 0"
nnUNetv2_train "$TASK_ID" 3d_fullres 0 -tr nnUNetTrainer_100Epochs

echo ""
echo "✅ 训练完成！结果保存于: ${nnUNet_results}"