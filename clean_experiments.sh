#!/bin/bash

# 基础路径
BASE_DIR="data/ForestDamages"

# 实验目录列表
EXPERIMENT_DIRS=(
    # 基础主动学习实验
    "active_learning_combinatorial"
    "active_learning_default"
    "active_learning_entropy"
    "active_learning_least_confidence"
    "active_learning_margin"
    "active_learning_random"
    "active_learning_rl_kl"
    "active_learning_rl_wasserstein"
    "active_learning_sor"
    "active_learning_ssc"
    "active_learning_wasserstein"
    
    # Cascade R-CNN 实验
    "active_learning_cascade_entropy"
    "active_learning_cascade_least_confidence"
    "active_learning_cascade_margin"
    "active_learning_cascade_mus_cdb"
    "active_learning_cascade_sor"
    "active_learning_cascade_ssc"
    
    # RetinaNet 实验
    "active_learning_retinanet_entropy"
    "active_learning_retinanet_least_confidence"
    "active_learning_retinanet_margin"
    "active_learning_retinanet_mus_cdb"
    "active_learning_retinanet_sor"
    "active_learning_retinanet_ssc"
    
    # Faster R-CNN 实验
    "active_learning_faster-rcnn_mus_cdb"
)

echo "开始清理主动学习实验目录..."
echo "总共需要清理 ${#EXPERIMENT_DIRS[@]} 个目录"

# 遍历每个实验目录并删除其中的文件
for dir in "${EXPERIMENT_DIRS[@]}"; do
    exp_path="$BASE_DIR/$dir"
    
    if [ -d "$exp_path" ]; then
        echo "正在清理目录: $exp_path"
        
        # 删除目录下的所有文件，但保留目录结构
        find "$exp_path" -type f -delete
        
        # 删除空的子目录，但保留主目录
        find "$exp_path" -mindepth 1 -type d -empty -delete
        
        echo "✓ 完成清理: $exp_path"
    else
        echo "⚠ 目录不存在，跳过: $exp_path"
    fi
done

echo ""
echo "========================================="
echo "所有主动学习实验目录清理完成!"
echo "总共处理了 ${#EXPERIMENT_DIRS[@]} 个目录"
echo "========================================="