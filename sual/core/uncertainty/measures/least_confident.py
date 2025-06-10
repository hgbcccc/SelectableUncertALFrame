import numpy as np
from typing import Dict
from mmdet.structures import DetDataSample

def least_confident_uncertainty(result: DetDataSample) -> Dict[str, float]:
    """基于最低置信度的不确定性计算
    
    Args:
        result (DetDataSample): MMDetection的检测结果
        
    Returns:
        Dict[str, float]: 最低置信度相关的不确定性度量
    """
    scores = result.pred_instances.scores.cpu().numpy()
    if len(scores) == 0:
        return {
            'least_confident': 0.0,
            'avg_least_confident': 0.0
        }
    
    # 计算1-最大置信度作为不确定性度量
    least_confident = 1 - np.max(scores)
    # 计算平均不确定性
    avg_least_confident = 1 - np.mean(scores)
    
    return {
        'least_confident': float(least_confident),
        'avg_least_confident': float(avg_least_confident)
    }