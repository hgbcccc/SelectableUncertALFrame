import numpy as np
from typing import Dict
from mmdet.structures import DetDataSample
from .utils import _calculate_sor


def calculate_sor(result: DetDataSample) -> Dict[str, float]:
    """计算空间遮挡率(SOR)"""
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    if len(bboxes) == 0:
        return {
            'max_sor': 0.0,
            'avg_sor': 0.0,
            'sum_sor': 0.0
        }

    sor_values = []
    for i in range(len(bboxes)):
        sor = _calculate_sor(bboxes[i], bboxes)  # 确保传递当前边界框和所有边界框
        sor_values.append(sor)

    return {
        'max_sor': float(np.max(sor_values)),
        'avg_sor': float(np.mean(sor_values)),
        'sum_sor': float(np.sum(sor_values))
    }