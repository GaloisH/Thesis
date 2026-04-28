#!/usr/bin/env bash
# preprocess.sh — 数据指纹提取与预处理
# 用法: bash preprocess.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${ROOT_DIR}/datasets"

export nnUNet_raw="${OUTPUT_DIR}/nnUNet_raw"
export nnUNet_preprocessed="${OUTPUT_DIR}/nnUNet_preprocessed"
export nnUNet_results="${OUTPUT_DIR}/nnUNet_results"

TASK_ID="101"
TASK_DIR="${nnUNet_raw}/Dataset101_Meningioma"

# 前置检查
if [[ ! -d "$TASK_DIR" ]] || [[ -z "$(ls -A "$TASK_DIR" 2>/dev/null)" ]]; then
    echo "❌ 未在 ${TASK_DIR} 检测到数据！"
    echo "   请首先运行: python src/prepare_data.py"
    exit 1
fi

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

echo "✅ 环境变量已设置:"
echo "   nnUNet_raw          = ${nnUNet_raw}"
echo "   nnUNet_preprocessed = ${nnUNet_preprocessed}"
echo "   nnUNet_results      = ${nnUNet_results}"

echo ""
echo "=============================================="
echo "数据指纹提取与预处理 (Plan & Preprocess)"
echo "=============================================="
echo ">> 执行命令: nnUNetv2_plan_and_preprocess -d ${TASK_ID} -c 2d --verify_dataset_integrity"

nnUNetv2_plan_and_preprocess \
    -d "$TASK_ID" \
    -c 2d \
    --verify_dataset_integrity

echo ""
echo "✅ 预处理完成！"