import numpy as np
from typing import Dict
from mmdet.structures import DetDataSample

# """基于熵的不确定性计算

# Args:
#     result (DetDataSample): MMDetection的检测结果
    
# Returns:
#     Dict[str, float]: 熵相关的不确定性度量
# """
# def entropy_uncertainty(result: DetDataSample) -> Dict[str, float]:
    # scores = result.pred_instances.scores.cpu().numpy()
    # # 归一化scores
    # norm_scores = scores / np.sum(scores)
    # entropy = -np.sum(norm_scores * np.log(norm_scores + 1e-10))

    # return {
    #     'entropy': float(entropy),
    #     'normalized_entropy': float(entropy / np.log(len(scores) + 1e-10))
    # }
def entropy_uncertainty(result: DetDataSample) -> Dict[str, float]:
    """基于all_scores的熵不确定性计算
    Args:
        result (DetDataSample): MMDetection的检测结果
    Returns:
        Dict[str, float]: 包含每个检测框熵、平均熵、全图类别分布熵等
    """
    # all_scores: [N, num_classes]
    all_scores = getattr(result.pred_instances, 'all_scores', None)
    if all_scores is None:
        raise ValueError("pred_instances中没有all_scores字段，请确认推理代码已正确输出所有类别概率。")
    all_scores = all_scores.cpu().numpy()  # [N, num_classes]
    eps = 1e-10
    # 1. 每个检测框的熵
    box_entropy = -np.sum(all_scores * np.log(all_scores + eps), axis=1)  # [N]
    mean_entropy = float(np.mean(box_entropy))
    # 2. 全图类别分布熵
    class_sum = np.sum(all_scores, axis=0)
    class_prob = class_sum / (np.sum(class_sum) + eps)
    class_entropy = -np.sum(class_prob * np.log(class_prob + eps))
    # 3. 归一化熵（以类别数为底）
    num_classes = all_scores.shape[1]
    normalized_entropy = float(mean_entropy / (np.log(num_classes) + eps))
    normalized_class_entropy = float(class_entropy / (np.log(num_classes) + eps))
    return {
        'box_entropy': box_entropy.tolist(), # 每个检测框的熵（box_entropy，list）
        'mean_entropy': mean_entropy, # 平均熵（mean_entropy，float）   
        'normalized_entropy': normalized_entropy, # 归一化熵（normalized_entropy，float）
        'class_entropy': float(class_entropy), # 全图类别分布熵（class_entropy，float）
        'normalized_class_entropy': normalized_class_entropy # 归一化全图类别分布熵（normalized_class_entropy，float）
    }