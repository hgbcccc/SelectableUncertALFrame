import numpy as np
from typing import Dict
from mmdet.structures import DetDataSample

def variance_uncertainty(result: DetDataSample) -> Dict[str, float]:
        """基于方差的不确定性计算
        
        Args:
            result (DetDataSample): MMDetection的检测结果
            
        Returns:
            Dict[str, float]: 方差相关的不确定性度量
        """
        scores = result.pred_instances.scores.cpu().numpy()
        uncertainties = 1 - scores
        
        return {
            'variance': float(np.var(uncertainties)),
            'std': float(np.std(uncertainties)),
            'cv': float(np.std(uncertainties) / (np.mean(uncertainties) + 1e-10))  # 变异系数
        }