import numpy as np
from typing import Dict, Optional
from mmdet.structures import DetDataSample
from .utils import _calculate_iou, _get_box_centers, _calculate_center_entropy, _calculate_center_variance

def box_uncertainty( 
                       result: DetDataSample, 
                       img_shape: Optional[tuple] = None) -> Dict[str, float]:
        """计算基于检测框的不确定性度量
        
        Args:
            result (DetDataSample): 检测结果
            img_shape (tuple, optional): 图片尺寸 (H, W)
            
        Returns:
            Dict[str, float]: 框相关的不确定性度量
        """
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        if len(bboxes) == 0:
            return {
                'iou_uncertainty': 0.0,
                'center_entropy': 0.0,
                'center_variance': 0.0
            }

        # 计算IoU不确定性
        total_iou = 0
        num_pairs = 0
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                iou = _calculate_iou(bboxes[i], bboxes[j])
                total_iou += iou
                num_pairs += 1
        
        iou_uncertainty = total_iou / (num_pairs + 1e-6)

        # 计算中心点
        centers = _get_box_centers(bboxes)
        
        # 标准化中心点坐标
        if img_shape is not None:
            centers[:, 0] /= img_shape[1]  # 宽度归一化
            centers[:, 1] /= img_shape[0]  # 高度归一化

        # 计算中心点熵和方差
        center_entropy = _calculate_center_entropy(centers)
        center_variance = _calculate_center_variance(centers)

        return {
            'iou_uncertainty': float(iou_uncertainty),
            'center_entropy': float(center_entropy),
            'center_variance': float(center_variance)
        }