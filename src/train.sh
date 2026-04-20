#!/usr/bin/env bash
# nnUNet 训练管线脚本
# 用法: bash train_nnunet.sh

# ─────────────────────────────────────────────
# 0. 路径配置
# ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${ROOT_DIR}/datasets"

export nnUNet_raw="${OUTPUT_DIR}/nnUNet_raw"
export nnUNet_preprocessed="${OUTPUT_DIR}/nnUNet_preprocessed"
export nnUNet_results="${OUTPUT_DIR}/nnUNet_results"

TASK_ID="101"
TASK_DIR="${nnUNet_raw}/Dataset101_Meningioma"

# ─────────────────────────────────────────────
# 1. 前置检查：数据是否存在
# ─────────────────────────────────────────────
if [[ ! -d "$TASK_DIR" ]] || [[ -z "$(ls -A "$TASK_DIR" 2>/dev/null)" ]]; then
    echo "❌ 未在 ${TASK_DIR} 检测到数据！"
    echo "   请首先运行: python src/prepare_data.py"
    exit 1
fi

# ─────────────────────────────────────────────
# 2. 创建目录 & 打印环境变量
# ─────────────────────────────────────────────
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

echo "✅ 环境变量已设置:"
echo "   nnUNet_raw          = ${nnUNet_raw}"
echo "   nnUNet_preprocessed = ${nnUNet_preprocessed}"
echo "   nnUNet_results      = ${nnUNet_results}"

# ─────────────────────────────────────────────
# 3. 阶段 1：数据指纹提取与预处理
# ─────────────────────────────────────────────
echo ""
echo "=============================================="
echo "阶段 1: 数据指纹提取与预处理 (Plan & Preprocess)"
echo "=============================================="
echo ">> 执行命令: nnUNetv2_plan_and_preprocess -d ${TASK_ID} -c 3d_fullres --verify_dataset_integrity"

nnUNetv2_plan_and_preprocess \
    -d "$TASK_ID" \
    -c 3d_fullres \
    --verify_dataset_integrity

# ─────────────────────────────────────────────
# 4. 阶段 2：启动网络训练
# ─────────────────────────────────────────────
echo ""
echo "=============================================="
echo "阶段 2: 启动网络训练 (Training)"
echo "=============================================="

nnUNetv2_train "$TASK_ID" 3d_fullres 0 -tr nnUNetTrainer_100epochs

echo ""
echo "✅ 训练管线全部完成！"