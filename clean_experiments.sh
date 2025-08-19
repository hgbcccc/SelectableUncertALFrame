#!/bin/bash

# 基础路径
BASE_DIR="data/ForestDamages"

# 实验目录列表
EXPERIMENT_DIRS=(
    "active_learning_basic"
    "active_learning_combinatorial"
    "active_learning_default"
    "active_learning_entropy"
    "active_learning_least_confid"
    "active_learning_ms"
    "active_learning_rl"
    "active_learning_rl_wasserstein"
    "active_learning_sor"
    "active_learning_ssc"
    "active_learning_wasserstein",
    "active_learning_cascade_margin",
    "active_learning_cascade_sor",
    "active_learning_cascade_ssc",
    "active_learning_cascade_entropy",
    "active_learning_cascade_least_confid",
    "active_learning_cascade_ms",
    "active_learning_cascade_rl",
    "active_learning_cascade_wasserstein",
    "active_learning_cascade_entropy",
    "active_learning_cascade_least_confid",
    "active_learning_cascade_ms",
    "active_learning_cascade_rl",
    "active_learning_cascade_wasserstein",
    "active_learning_cascade_entropy",
    "active_learning_cascade_least_confid",
    "active_learning_cascade_ms",
    "active_learning_cascade_rl",
    "active_learning_cascade_wasserstein",
    "active_learning_retinanet_margin",
    "active_learning_retinanet_sor",
    "active_learning_retinanet_ssc",
    "active_learning_retinanet_entropy",
    "active_learning_retinanet_least_confid",
    "active_learning_retinanet_ms",
    "active_learning_retinanet_rl",
    "active_learning_retinanet_wasserstein",
    
    
    )

# 遍历每个实验目录并删除其中的文件
for dir in "${EXPERIMENT_DIRS[@]}"; do
    exp_path="$BASE_DIR/$dir"
    echo "清理目录: $exp_path"
    
    # 删除目录下的所有文件，但保留目录结构
    find "$exp_path" -type f -delete
    
    # 如果需要也删除子目录，但保留主目录，可以用:
    find "$exp_path" -mindepth 1 -type d -empty -delete
    
    echo "完成清理: $exp_path"
done

echo "所有实验目录清理完成"