# import numpy as np
# import torch
# from typing import Dict
# from mmdet.structures import DetDataSample

# def MUS_CDB(result: DetDataSample) -> Dict[str, float]:
#     """
#     实现MUS-CDB主动学习采样策略，计算每个边界框的混合不确定性得分和类别分布平衡权重
    
#     参数:
#     result (DetDataSample): 模型推理结果，包含预测边界框、置信度和标签信息
    
#     返回:
#     Dict[str, float]: 每个边界框的最终得分，格式为 {bbox_index: final_score}
#     """
#     # 获取预测实例
#     pred_instances = result.pred_instances
    
#     # 1. 提取必要信息
#     scores = pred_instances.scores.cpu().numpy()  # 每个bbox的最大置信度
#     all_scores = pred_instances.all_scores.cpu().numpy()  # 每个bbox的所有类别置信度
#     bboxes = pred_instances.bboxes.cpu().numpy()  # 边界框坐标
#     labels = pred_instances.labels.cpu().numpy()  # 预测的类别标签
    
#     # 2. 设置超参数
#     confidence_threshold = 0.5  # 用于筛选高置信度目标的阈值
    
#     # 3. 计算图像级不确定性
#     high_confidence_indices = np.where(scores >= confidence_threshold)[0]
#     if len(high_confidence_indices) > 0:
#         high_confidence_scores = scores[high_confidence_indices]
#         image_uncertainty = 1.0 - np.mean(high_confidence_scores)
#     else:
#         # 如果没有高置信度目标，图像不确定性设为1
#         image_uncertainty = 1.0
    
#     # 4. 计算每个目标的不确定性
#     object_uncertainties = []
#     for score_dist in all_scores:
#         # 计算类别概率分布的熵
#         entropy = -np.sum(score_dist * np.log2(score_dist + 1e-10))
#         object_uncertainties.append(entropy)
#     object_uncertainties = np.array(object_uncertainties)
    
#     # 5. 计算混合不确定性得分 (MUS)
#     mus_scores = image_uncertainty * object_uncertainties
    
#     # 6. 计算类别分布平衡权重 (CDB)
#     unique_labels, label_counts = np.unique(labels, return_counts=True)
#     total_instances = len(labels)
    
#     # 计算每个类别的频率和逆频率
#     label_freq = label_counts / total_instances
#     label_inverse_freq = 1.0 - label_freq
    
#     # 对逆频率应用softmax以获得类别偏好权重
#     exp_inverse_freq = np.exp(label_inverse_freq)
#     cdb_weights = exp_inverse_freq / np.sum(exp_inverse_freq)
    
#     # 创建标签到CDB权重的映射
#     label_to_cdb = {label: weight for label, weight in zip(unique_labels, cdb_weights)}
    
#     # 为每个边界框分配CDB权重
#     bbox_cdb_weights = np.array([label_to_cdb[label] for label in labels])
    
#     # 7. 计算最终得分：MUS和CDB的加权组合
#     alpha = 0.7  # MUS权重
#     beta = 0.3   # CDB权重
#     final_scores = alpha * mus_scores + beta * bbox_cdb_weights
    
#     # 8. 返回结果
#     return {str(i): float(score) for i, score in enumerate(final_scores)}   
# 
# 
# 
# 
#
# 



import numpy as np
from typing import Dict
from mmdet.structures import DetDataSample

def mus_uncertainty(
    result: DetDataSample,
    theta: float = 0.1  # 论文推荐阈值θ=0.1
) -> Dict[str, float]:
    """
    计算混合不确定性得分MUS，公式：φ_ij = φ_i^I * φ_ij^O
    返回每个边界框的MUS得分，键为"bbox_<index>"
    """
    if not hasattr(result, 'pred_instances') or len(result.pred_instances) == 0:
        return {}
    
    pred = result.pred_instances
    scores = pred.scores.cpu().numpy()         # 单类别最大置信度
    all_scores = pred.all_scores.cpu().numpy()   # 所有类别置信度矩阵
    N = len(scores)
    
    # 1. 计算图像级不确定性 φ_i^I
    high_conf_mask = scores > theta
    if np.any(high_conf_mask):
        avg_high_conf = np.mean(scores[high_conf_mask])
        img_uncertainty = 1.0 - avg_high_conf
    else:
        img_uncertainty = 1.0  # 无高置信目标时设为最大值
    
    # 2. 计算目标级不确定性 φ_ij^O（熵）
    eps = 1e-10  # 数值稳定性
    obj_uncertainties = -np.sum(all_scores * np.log(all_scores + eps), axis=1)
    
    # 3. 混合得分
    mus_scores = img_uncertainty * obj_uncertainties
    
    # 返回格式：{bbox索引: MUS得分}
    return {f"bbox_{i}": float(mus_scores[i]) for i in range(N)}