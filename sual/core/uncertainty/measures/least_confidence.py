import numpy as np
from typing import Dict
from mmdet.structures import DetDataSample

def least_confidence_uncertainty(result: DetDataSample) -> Dict[str, float]:
    """基于最低置信度的不确定性计算
    
    Args:
        result (DetDataSample): MMDetection的检测结果
        
    Returns:
        Dict[str, float]: 最低置信度相关的不确定性度量，基于每个框的类别概率分布
            - least_confidence: 最大不确定性（主要指标）
            - mean_least_confidence: 平均不确定性
    """
    print("Computing least confidence uncertainty")
    
    # 检查是否有all_scores属性
    if not hasattr(result.pred_instances, 'all_scores'):
        print("No all_scores found in pred_instances")
        return {
            'least_confidence': 0.0,
            'mean_least_confidence': 0.0
        }
    
    # 获取每个框对所有类别的预测概率（已经是概率分布）
    all_scores = result.pred_instances.all_scores.cpu().numpy()  # shape: [num_boxes, num_classes]
    
    # print(f"All scores shape: {all_scores.shape}")
    # print(f"All scores range: [{all_scores.min()}, {all_scores.max()}]")
    
    if len(all_scores) == 0:
        print("Empty all_scores array")
        return {
            'least_confidence': 0.0,
            'mean_least_confidence': 0.0
        }
    
    # 每个框的最高类别概率
    max_class_probs = np.max(all_scores, axis=1)  # shape: [num_boxes]
    
    # 每个框的不确定性 (1 - 最高概率)
    box_uncertainties = 1 - max_class_probs  # shape: [num_boxes]
    
    # 计算不同的图片级不确定性指标
    # 1. 平均不确定性：所有框的不确定性的平均值
    mean_least_confidence = float(np.mean(box_uncertainties))
    
    # 2. 最大不确定性：最不确定的那个框的不确定性
    least_confidence = float(np.max(box_uncertainties))
    
    # print(f"Computed metrics: mean_least_confidence={mean_least_confidence}, least_confidence={least_confidence}")
    
    # 确保返回的键名与配置文件中的期望一致
    return {
        'least_confidence': least_confidence,          # 最大不确定性（主要指标）
        'mean_least_confidence': mean_least_confidence # 平均不确定性（与配置文件匹配）
    }