from typing import Dict, List, Optional, Tuple
import numpy as np


'''
    Recalculate crown count scores utility functions.

    This module contains helper functions for recalculating crown count scores, including:
    - Recalculating crown count scores
'''

# 重新计算未标注池中每张图片的树冠控制系数
def recalculate_crown_count_scores(unlabeled_pool_results, unlabeled_pool_results_uncertainty):
    """重新计算未标注池中每张图片的树冠控制系数
    
    基于当前未标注池的批次统计信息，重新计算每张图片的树冠控制系数，
    并更新到不确定性结果中。
    
    Args:
        unlabeled_pool_results: 未标注池的推理结果
        unlabeled_pool_results_uncertainty: 未标注池的不确定性结果
        
    Returns:
        更新后的不确定性结果
    """
    # 计算未标注池的批次统计信息
    unlabeled_batch_stats = {
        'batch_statistics': {
            'mean': 0.0,
            'std': 0.0,
            'min': float('inf'),
            'max': float('-inf')
        }
    }
    
    # 收集所有图片的检测框数量
    crown_counts = []
    for img_name, img_data in unlabeled_pool_results.items():
        if 'result' in img_data and hasattr(img_data['result'], '_pred_instances'):
            bboxes = img_data['result']._pred_instances.bboxes
            crown_count = len(bboxes)
            crown_counts.append(crown_count)
            unlabeled_batch_stats['batch_statistics']['min'] = min(
                unlabeled_batch_stats['batch_statistics']['min'], crown_count)
            unlabeled_batch_stats['batch_statistics']['max'] = max(
                unlabeled_batch_stats['batch_statistics']['max'], crown_count)
    
    # 计算均值和标准差
    if crown_counts:
        unlabeled_batch_stats['batch_statistics']['mean'] = sum(crown_counts) / len(crown_counts)
        unlabeled_batch_stats['batch_statistics']['std'] = np.std(crown_counts) if len(crown_counts) > 1 else 0.0
    else:
        unlabeled_batch_stats['batch_statistics']['min'] = 0
        unlabeled_batch_stats['batch_statistics']['max'] = 0
    
    # 重新计算每张图片的树冠控制系数
    for img_name, img_data in unlabeled_pool_results.items():
        if 'result' in img_data and hasattr(img_data['result'], '_pred_instances'):
            # 获取检测结果
            result = img_data['result']
            bboxes = result._pred_instances.bboxes
            labels = result._pred_instances.labels
            
            # 计算树冠数量
            crown_count = len(bboxes)
            
            # 使用批次统计信息计算树冠控制系数
            stats = unlabeled_batch_stats['batch_statistics']
            n_batch = stats['mean']
            sigma_batch = stats['std']
            
            # 放宽边界阈值
            upper_threshold = stats['max'] * 2.0
            lower_threshold = stats['min'] * 0.3
            
            # 计算基础高斯得分
            normalized_diff = (crown_count - n_batch) / (sigma_batch + 1e-6)
            base_score = np.exp(-0.5 * normalized_diff ** 2)
            
            # 含过渡和平滑优化
            transition_width = 25
            slope = 8.0
            
            # 上界平滑过渡
            upper_smooth = 1 / (1 + np.exp((crown_count - upper_threshold + transition_width/2)/transition_width*slope))
            
            # 下界平滑过渡  
            lower_smooth = 1 / (1 + np.exp((-crown_count + lower_threshold + transition_width/2)/transition_width*slope))
            
            # 综合得分
            crown_count_score = base_score * upper_smooth * lower_smooth
            
            # 确保分数不低于最小值
            # min_score = 0.00
            # crown_count_score = max(min_score, crown_count_score)
            
            # 更新不确定性结果中的树冠控制系数
            if img_name in unlabeled_pool_results_uncertainty and 'uncertainty' in unlabeled_pool_results_uncertainty[img_name]:
                unlabeled_pool_results_uncertainty[img_name]['uncertainty']['crown_count_score'] = crown_count_score
    print(f"n_batch: {n_batch}, sigma_batch: {sigma_batch}")
    return unlabeled_pool_results_uncertainty
