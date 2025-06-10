#!/bin/bash
# custom_config/中运行
# 切换到项目根目录
cd ..

# 基础数据集路径
BASE_DATA_DIR="data/ForestDamages"

# 实验配置
EXPERIMENTS=(
    "Default"
    "RL_KL"
    "RL_Wasserstein"
    "Wasserstein"
)

# 为每个实验准备数据集
for exp in "${EXPERIMENTS[@]}"; do
    OUTPUT_DIR="${BASE_DATA_DIR}/active_learning_${exp,,}"  # 转换为小写
    echo "准备数据集: ${exp} -> ${OUTPUT_DIR}"
    
    python sual/tools/prepare_dataset/prepare_active_dataset.py \
        ${BASE_DATA_DIR} \
        ${OUTPUT_DIR} \
        --train-ratio 0.04 \
        --val-ratio 0.01 \
        --seed 42
done