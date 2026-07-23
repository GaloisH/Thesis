#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=$SCRIPT_DIR
OUTPUT_DIR="${ROOT_DIR}/datasets"

export nnUNet_raw="${OUTPUT_DIR}/nnUNet_raw"
export nnUNet_preprocessed="${OUTPUT_DIR}/nnUNet_preprocessed"
export nnUNet_results="${OUTPUT_DIR}/nnUNet_results"

# WandB
export nnUNet_wandb_enabled=1
export nnUNet_wandb_project="nnUNet_Meningitis"

TASK_ID="002"
CONFIG="3d_fullres"
TRAINER="nnUNetTrainer_100epochs"

echo "=============================================="
echo "Environment"
echo "=============================================="
echo "nnUNet_raw          = ${nnUNet_raw}"
echo "nnUNet_preprocessed = ${nnUNet_preprocessed}"
echo "nnUNet_results      = ${nnUNet_results}"
echo ""

for FOLD in 0 1 2 3 4
do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    export WANDB_NAME="Task${TASK_ID}_${CONFIG}_fold${FOLD}_${TIMESTAMP}"

    echo "=============================================="
    echo "Training Fold ${FOLD}"
    echo "WANDB_NAME=${WANDB_NAME}"
    echo "=============================================="

    nnUNetv2_train \
        "${TASK_ID}" \
        "${CONFIG}" \
        "${FOLD}" \
        -tr "${TRAINER}" \

    echo "Fold ${FOLD} Finished"
    echo ""
done

echo "=============================================="
echo "All Training Finished"
echo "=============================================="