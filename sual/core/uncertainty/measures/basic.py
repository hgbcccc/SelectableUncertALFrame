import numpy as np
from typing import Dict, Union, List, Optional
from mmdet.structures import DetDataSample

def basic_uncertainty(result: DetDataSample) -> Dict[str, float]:
        """基础的不确定性计算(1-置信度)
    
    Args:
        result (DetDataSample): MMDetection的检测结果
            - result.pred_instances: 预测实例的信息
                - scores: 每个检测框的置信度 [N,]
                - bboxes: 每个检测框的坐标 [N, 4]
                - labels: 每个检测框的类别 [N,]
            
    Returns:
        Dict[str, float]: 包含max/sum/avg三种不确定性度量的字典
    """
        """基础的不确定性计算(1-置信度)
        
        Args:
            result (DetDataSample): MMDetection的检测结果
            
        Returns:
            Dict[str, float]: 包含max/sum/avg三种不确定性度量的字典
        """
        scores = result.pred_instances.scores.cpu().numpy()
        uncertainties = 1 - scores
        
        return {
            'max_uncertainty': float(np.max(uncertainties)),
            'sum_uncertainty': float(np.sum(uncertainties)),
            'avg_uncertainty': float(np.mean(uncertainties))
        }