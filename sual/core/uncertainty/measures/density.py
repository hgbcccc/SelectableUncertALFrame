import numpy as np
from typing import Dict
from mmdet.structures import DetDataSample

def density_uncertainty(result: DetDataSample) -> Dict[str, float]:
        """基于密度的不确定性计算
        
        Args:
            result (DetDataSample): MMDetection的检测结果
            
        Returns:
            Dict[str, float]: 密度相关的不确定性度量
        """
        scores = result.pred_instances.scores.cpu().numpy()
        uncertainties = 1 - scores
        
        # 计算高不确定性(>0.5)的比例
        high_uncertainty_ratio = np.mean(uncertainties > 0.5)
        
        return {
            'high_uncertainty_ratio': float(high_uncertainty_ratio),
            'uncertainty_density': float(len(uncertainties) / (np.max(uncertainties) - np.min(uncertainties) + 1e-10))
        }