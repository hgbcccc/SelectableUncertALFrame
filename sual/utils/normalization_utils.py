from typing import Dict, List, Optional, Tuple
import numpy as np


'''
    Normalization utility functions.

    This module contains helper functions for normalization process, including:
    - Uncertainty score normalization
    - Training statistics normalization
'''

# 标准化不确定性分数
def normalize_uncertainty_scores(unlabeled_pool_results):
    # Step 1: 收集所有指标的全局值
    metric_values = {}
    
    # 遍历所有图片收集指标
    for img_data in unlabeled_pool_results.values():
        if 'uncertainty' not in img_data:
            continue
        for metric, value in img_data['uncertainty'].items():
            if isinstance(value, (int, float)):
                metric_values.setdefault(metric, []).append(value)

    # Step 2: 计算各指标的min-max范围
    metric_ranges = {
        metric: {'min': min(values), 'max': max(values)}
        for metric, values in metric_values.items()
    }

    # Step 3: 创建标准化后的数据结构
    normalized_results = {}
    
    for img_name, img_data in unlabeled_pool_results.items():
        if 'uncertainty' not in img_data:
            normalized_results[img_name] = img_data
            continue

        normalized_metrics = {}
        for metric, value in img_data['uncertainty'].items():
            if metric not in metric_ranges:
                normalized_metrics[metric] = value
                continue

            v_min = metric_ranges[metric]['min']
            v_max = metric_ranges[metric]['max']
            
            # 处理除零情况
            if v_max == v_min:
                normalized = 0.5  # 当所有值相同时设为中间值
            else:
                normalized = (value - v_min) / (v_max - v_min)
                normalized = max(0.0, min(1.0, normalized))  # 确保在[0,1]范围内

            normalized_metrics[metric] = round(normalized, 4)

        # 保持原有数据结构
        normalized_results[img_name] = {
            **img_data,
            'uncertainty_normalized': normalized_metrics
        }

    return normalized_results

# 标准化训练集统计信息
def normalize_train_stats(train_stats):
    """对训练集统计信息进行全局标准化
    
    将训练集中的特征分布标准化到[0,1]范围内，便于与未标注样本进行比较
    
    Args:
        train_stats: 训练集统计信息字典
        
    Returns:
        标准化后的训练集统计信息字典
    """
    if 'features' not in train_stats:
        return train_stats
    
    normalized_train_stats = {
        'features': {},
        'batch_statistics': train_stats.get('batch_statistics', {})
    }
    
    # 对每个特征进行标准化
    for feature_name, feature_stats in train_stats['features'].items():
        if 'distribution' not in feature_stats or not feature_stats['distribution']:
            normalized_train_stats['features'][feature_name] = feature_stats
            continue
        
        # 获取分布的最小值和最大值
        distribution = feature_stats['distribution']
        v_min = min(distribution)
        v_max = max(distribution)
        
        # 处理除零情况
        if v_max == v_min:
            normalized_distribution = [0.5] * len(distribution)  # 当所有值相同时设为中间值
        else:
            # 标准化分布
            normalized_distribution = [
                max(0.0, min(1.0, (x - v_min) / (v_max - v_min)))
                for x in distribution
            ]
        
        # 计算标准化后的均值和标准差
        normalized_mean = sum(normalized_distribution) / len(normalized_distribution)
        normalized_std = np.std(normalized_distribution) if len(normalized_distribution) > 1 else 0.0
        
        # 保存标准化后的统计信息
        normalized_train_stats['features'][feature_name] = {
            'mean': normalized_mean,
            'std': normalized_std,
            'distribution': normalized_distribution,
            'min': 0.0,  # 标准化后的最小值
            'max': 1.0   # 标准化后的最大值
        }
    
    return normalized_train_stats