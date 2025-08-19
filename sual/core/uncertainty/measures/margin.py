import numpy as np
from typing import Dict
import torch
from mmdet.structures import DetDataSample

def margin_uncertainty(result: DetDataSample) -> Dict[str, float]:
    """基于边缘采样的不确定性计算
    
    对每个检测框，计算其最高类别概率与次高类别概率的差值（margin），
    用于评估模型在类别预测上的不确定性。margin越小，表示模型对前两个最可能的类别的区分越不确定。
    
    Args:
        result (DetDataSample): MMDetection的检测结果，包含每个检测框对所有类别的预测概率
            - pred_instances.all_scores: 形状为[num_boxes, num_classes]的张量，
              表示每个检测框对所有类别的概率分布（已经过softmax）
    
    Returns:
        Dict[str, float]: 边缘采样相关的不确定性度量
            - margin: 所有检测框的平均不确定性 (1-margin的均值)
            - mean_margin: 所有检测框的平均margin值（用于分析）
            - min_margin: 所有检测框中的最小margin值（最不确定的情况）
    
    Notes:
        1. 输入的all_scores应该已经是概率分布（和为1的概率值）
        2. 对每个框找出最高和次高的类别概率
        3. 计算margin（最高概率-次高概率）
        4. 使用1-margin作为不确定性指标，使其与其他不确定性度量保持一致（值越大表示越不确定）
        5. 如果没有检测框或只有一个类别，返回默认值
    
    Example:
        >>> result = DetDataSample(...)  # 包含检测结果
        >>> uncertainty = margin_uncertainty(result)
        >>> print(f"平均不确定性: {uncertainty['margin']:.3f}")
        >>> print(f"最大不确定性: {1 - uncertainty['min_margin']:.3f}")
    """
    # 检查是否有all_scores属性
    if not hasattr(result.pred_instances, 'all_scores'):
        return {'margin': 0.0, 'mean_margin': 0.0, 'min_margin': 0.0}
    
    # 转换为numpy数组进行计算
    all_scores = result.pred_instances.all_scores.cpu().numpy()
    num_boxes, num_classes = all_scores.shape
    
    # 处理边界情况：无检测框或只有一个类别
    if num_boxes == 0 or num_classes == 1:
        return {'margin': 0.0, 'mean_margin': 0.0, 'min_margin': 0.0}
    
    # 使用partition高效获取每个框的最高和次高概率
    # partition比完整排序更快，因为我们只需要最后两个值
    top2_probs = np.partition(all_scores, kth=-2, axis=1)[:, -2:]
    max_probs = top2_probs[:, -1]    # 每个框的最高类别概率
    second_probs = top2_probs[:, -2]  # 每个框的次高类别概率
    
    # 计算margin和不确定性指标
    margins = max_probs - second_probs  # margin越大表示越确定
    uncertainties = 1 - margins         # 转换为不确定性指标，越大表示越不确定
    
    return {
        'margin': float(np.mean(uncertainties)),  # 平均不确定性（主要指标）
        'mean_margin': float(np.mean(margins)),   # 平均margin（用于分析）
        'min_margin': float(np.min(margins)),     # 最小margin（最不确定的情况）
    }