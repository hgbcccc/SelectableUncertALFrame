import numpy as np
from typing import Dict
from mmdet.structures import DetDataSample

def quantile_uncertainty(result: DetDataSample) -> Dict[str, float]:
        """基于分位数的不确定性计算
        
        Args:
            result (DetDataSample): MMDetection的检测结果
            
        Returns:
            Dict[str, float]: 分位数相关的不确定性度量
        """
        scores = result.pred_instances.scores.cpu().numpy()
        uncertainties = 1 - scores
        
        return {
            'q25': float(np.percentile(uncertainties, 25)),
            'q50': float(np.percentile(uncertainties, 50)),  # 中位数
            'q75': float(np.percentile(uncertainties, 75)),
            'iqr': float(np.percentile(uncertainties, 75) - np.percentile(uncertainties, 25))  # 四分位距
        }