import numpy as np
from typing import Dict, Optional
from mmdet.structures import DetDataSample
from .utils import _calculate_sor, _get_box_centers

def calculate_ssc(result: DetDataSample, batch_stats: Optional[Dict] = None, alpha: float = 0.3) -> Dict[str, float]:
    """计算空间结构复杂度(SSC)
    
    Args:
        result (DetDataSample): 检测结果
        batch_stats (Optional[Dict]): 批次统计信息，包含mean和std
        alpha (float, optional): top-k选择的超参数. Defaults to 0.3.
    
    Returns:
        Dict[str, float]: SSC各项指标的计算结果
    """
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()
    
    if len(bboxes) == 0:
        return {
            'ssc_score': 0.0,
            'occlusion_score': 0.0,
            'crown_count_score': 0.0,
            'diversity_score': 0.0,
            'area_var_score': 0.0,
            'density_var_score': 0.0
        }
    
    # 1. 计算遮挡系数 (使用动态top-k)
    k = max(1, int(alpha * len(bboxes)))  # 动态确定k值
    sor_values = []
    for i in range(len(bboxes)):
        sor = _calculate_sor(bboxes[i], bboxes)
        sor_values.append(sor)
    
    # 取top-k最大的SOR值的平均作为遮挡系数
    sorted_sor = sorted(sor_values, reverse=True)
    occlusion_score = np.mean(sorted_sor[:k])
    
    # 2. 计算树冠数量控制系数
    # 获取当前图像中的树冠数量
    crown_count = len(bboxes)
    
    # 使用批次统计信息或默认值
    if batch_stats and 'mean' in batch_stats and 'std' in batch_stats:
        # 使用提供的批次统计信息（来自主动学习每轮对未标注样本的推理结果）
        # 这些统计信息反映了当前未标注数据集中的候选框数量分布
        n_batch = batch_stats['mean']  # 未标注数据集中的平均候选框数量
        sigma_batch = max(1.0, batch_stats['std'])  # 确保标准差不为0，避免除零错误
    else:
        # 使用默认统计值（当没有提供批次统计信息时）
        # 统计森林损坏数据集的标注统计结果：
        # 整个数据集标注统计结果：
        # 平均每张图像标注数：56.66
        # 标注数量标准差：34.75
        # 建议默认值配置：n_batch=57, sigma_batch=35

        n_batch = 57  # 默认批次均值
        sigma_batch = 35  # 默认批次标准差
        print(f"未提供批次统计信息，使用默认统计值：均值={n_batch}, 标准差={sigma_batch}")
    
    # 计算正态分布概率密度值
    # 这里使用高斯函数计算当前样本中的候选框数量与批次平均值的偏差程度
    # 当候选框数量接近批次平均值时，得分接近1；偏离越远，得分越低
    # 这种机制在保证样本信息量的同时，避免了选择标注成本过高的样本
    normalized_diff = (crown_count - n_batch) / sigma_batch
    crown_count_score = np.exp(-0.5 * normalized_diff ** 2)
    
    # 确保分数在0-1范围内
    crown_count_score = min(1.0, crown_count_score)
    
    # 3. 计算类别多样性系数
    # 根据公式 e_i = -ln(1 + |C_i|) × Σ[p(c_k)log p(c_k)]，
    # 其中C_i是影像i包含的树冠类型集合，c_k是第k类树冠的占比。
    # 这个系数不仅考虑了类别数量（通过ln(1 + |C_i|)项），
    # 还考虑了各类别分布的均匀性（通过熵项Σ[p(c_k)log p(c_k)]），
    # 从而更准确地反映了树冠类型的多样性。
    #  统计每个类别的数量
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    # 计算每个类别的占比
    label_proportions = label_counts / len(labels)
    # 计算类别数量影响因子
    category_factor = -np.log(1 + len(unique_labels))
    # 计算熵值
    entropy = np.sum(label_proportions * np.log(label_proportions + 1e-6))
    # 计算最终的多样性得分
    diversity_score = category_factor * entropy
    
    # 4. 计算边界框面积变异系数
    areas = [(box[2]-box[0])*(box[3]-box[1]) for box in bboxes]
    area_mean = np.mean(areas)
    
    # 计算面积方差 σ²_i,A
    area_variance = np.sum([(area - area_mean)**2 for area in areas]) / (len(areas) + 1e-6)
    
    # 归一化参数 σ²_A0 (动态计算)
    # 计算理论上可能的最大面积方差
    # 假设最极端情况：一半边界框面积为0，一半为2*area_mean
    # 此时方差最大，为 area_mean²
    sigma_a0_squared = max(area_mean**2, area_variance)  # 使用理论最大值和实际值中的较大者
    
    # 计算边界框面积变异系数: a_i = (σ²_i,A / σ²_A0)
    # 确保值在0到1之间
    #
    area_var_score = area_variance / (sigma_a0_squared + 1e-6)
    area_var_score = area_var_score + 1

    # 5. 计算局部空间密度变异系数
    centers = _get_box_centers(bboxes)
    distances = []
    for i in range(len(centers)):
        dist = np.sqrt(np.sum((centers - centers[i])**2, axis=1))
        dist.sort()
        # 取最近的k个邻居（不包括自己）的平均距离
        local_density = np.mean(dist[1:min(k+1, len(dist))])
        distances.append(local_density)
    
    density_mean = np.mean(distances)
    density_std = np.std(distances)
    density_var_score = density_std / (density_mean + 1e-6)
    density_var_score = min(1.0, density_var_score)  # 截断到1
    
    # 综合得分：所有指标的加权平均
    weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # 各指标的权重
    scores = [occlusion_score, crown_count_score, diversity_score, 
              area_var_score, density_var_score]
    ssc_score = np.sum(np.array(weights) * np.array(scores))
    
    return {
        'ssc_score': float(ssc_score),
        'occlusion_score': float(occlusion_score),
        'crown_count_score': float(crown_count_score),
        'diversity_score': float(diversity_score),
        'area_var_score': float(area_var_score),
        'density_var_score': float(density_var_score)
    }