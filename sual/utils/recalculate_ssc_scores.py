from typing import Dict, List, Optional, Tuple
import numpy as np
from mmdet.structures import DetDataSample
from sklearn.neighbors import KernelDensity


def recalculate_ssc_scores(unlabeled_pool_results, unlabeled_pool_results_uncertainty):
    """重新计算未标注池中每张图片的所有SSC指标
    
    基于当前未标注池的全局统计信息，重新计算每张图片的所有SSC指标：
    1. 遮挡系数 (occlusion_score)
    2. 树冠控制系数 (crown_count_score)
    3. 类别多样性系数 (diversity_score)
    4. 边界框面积变异系数 (area_var_score)
    5. 局部空间密度变异系数 (density_var_score)
    6. 综合SSC分数 (ssc_score)
    
    Args:
        unlabeled_pool_results: 未标注池的推理结果
        unlabeled_pool_results_uncertainty: 未标注池的不确定性结果
        
    Returns:
        更新后的不确定性结果
    """
    # 第一步：收集全局统计信息
    global_stats = collect_global_statistics(unlabeled_pool_results)
    
    # 第二步：重新计算每张图片的所有SSC指标
    processed_count = 0
    total_count = len(unlabeled_pool_results)
    
    for img_name, img_data in unlabeled_pool_results.items():
        if 'result' in img_data and hasattr(img_data['result'], 'pred_instances'):
            result = img_data['result']
            
            # 重新计算所有SSC指标
            occlusion_score = calculate_occlusion_score(result)
            crown_count_score = calculate_crown_count_score(result, global_stats)
            diversity_score = calculate_diversity_score(result)
            area_var_score = calculate_area_var_score(result, global_stats)
            density_var_score = calculate_density_var_score(result)
            
            # 计算综合SSC分数（使用默认权重）
            weights = [2, 1, 2, 1, 1]  # 默认权重
            ssc_score = float(np.dot(weights, [
                occlusion_score,
                crown_count_score,
                diversity_score,
                area_var_score,
                density_var_score
            ]))
            
            # 更新不确定性结果
            if img_name in unlabeled_pool_results_uncertainty and 'uncertainty' in unlabeled_pool_results_uncertainty[img_name]:
                unlabeled_pool_results_uncertainty[img_name]['uncertainty'].update({
                    'occlusion_score': occlusion_score,
                    'crown_count_score': crown_count_score,
                    'diversity_score': diversity_score,
                    'area_var_score': area_var_score,
                    'density_var_score': density_var_score,
                    'ssc_score': ssc_score
                })
        
        processed_count += 1
        
        # 每处理100张图片输出一次进度
        if processed_count % 100 == 0 or processed_count == total_count:
            print(f"SSC计算进度: [{processed_count}/{total_count}] "
                  f"occlusion: {occlusion_score:.3f} "
                  f"crown: {crown_count_score:.3f} "
                  f"diversity: {diversity_score:.3f} "
                  f"area_var: {area_var_score:.3f} "
                  f"density: {density_var_score:.3f} "
                  f"ssc: {ssc_score:.3f}")
    
    print(f"全局统计信息:")
    print(f"  平均标注数: {global_stats['mean_annotation_count']:.2f}")
    print(f"  标注数标准差: {global_stats['std_annotation_count']:.2f}")
    print(f"  最大方差 (sigma_a0): {global_stats['max_variance']:.2f}")
    print(f"  平均方差: {global_stats['mean_variance']:.2f}")
    
    return unlabeled_pool_results_uncertainty


def collect_global_statistics(unlabeled_pool_results):
    """收集全局统计信息"""
    all_areas = []
    all_variances = []
    all_annotation_counts = []
    
    for img_name, img_data in unlabeled_pool_results.items():
        if 'result' in img_data and hasattr(img_data['result'], 'pred_instances'):
            result = img_data['result']
            bboxes = result.pred_instances.bboxes.cpu().numpy()
            
            if len(bboxes) > 0:
                # 计算面积
                areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                all_areas.extend(areas)
                
                # 计算方差
                area_var = np.var(areas)
                all_variances.append(area_var)
                
                # 计算标注数量
                all_annotation_counts.append(len(bboxes))
    
    if not all_areas:
        print("警告: 没有找到有效的面积数据，使用默认值")
        return {
            'mean_annotation_count': 150,
            'std_annotation_count': 50,
            'max_variance': 1000000,
            'mean_variance': 500000
        }
    
    # 计算全局统计
    global_stats = {
        'mean_annotation_count': float(np.mean(all_annotation_counts)),
        'std_annotation_count': float(np.std(all_annotation_counts)),
        'max_variance': float(np.max(all_variances)),  # 实际最大方差
        'mean_variance': float(np.mean(all_variances))
    }
    
    return global_stats


def calculate_crown_count_score(result: DetDataSample, global_stats: Dict) -> float:
    """计算树冠控制系数"""
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    crown_count = len(bboxes)
    
    # 使用全局统计信息
    n_batch = global_stats['mean_annotation_count']
    sigma_batch = global_stats['std_annotation_count']
    
    # 设置边界阈值
    upper_threshold = n_batch + 2 * sigma_batch
    lower_threshold = n_batch - 2 * sigma_batch
    
    # 计算基础高斯得分
    normalized_diff = (crown_count - n_batch) / (sigma_batch + 1e-6)
    base_score = np.exp(-0.5 * normalized_diff ** 2)
    
    # 平滑过渡
    transition_width = 25
    slope = 8.0
    
    # 上下界平滑过渡
    upper_smooth = 1 / (1 + np.exp((crown_count - upper_threshold + transition_width/2)/transition_width*slope))
    lower_smooth = 1 / (1 + np.exp((-crown_count + lower_threshold + transition_width/2)/transition_width*slope))
    
    # 综合得分
    crown_count_score = base_score * upper_smooth * lower_smooth
    
    return float(crown_count_score)


def calculate_area_var_score(result: DetDataSample, global_stats: Dict) -> float:
    """计算边界框面积变异系数"""
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    
    if len(bboxes) == 0:
        return 1.0
    
    # 计算当前图片的面积方差
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    area_var = np.var(areas)
    
    # 使用全局最大方差作为sigma_a0
    sigma_a0 = global_stats['max_variance']
    
    if sigma_a0 > 0:
        normalized_var = area_var / sigma_a0
        area_var_score = 1 + normalized_var
    else:
        area_var_score = 1.0
    
    return float(area_var_score)


def calculate_occlusion_score(result: DetDataSample) -> float:
    """计算遮挡系数"""
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    
    if len(bboxes) == 0:
        return 0.0
    
    # 计算每个边界框的OR（重叠率）
    or_values = []
    for i in range(len(bboxes)):
        # 计算当前边界框与其他边界框的重叠
        current_bbox = bboxes[i]
        overlaps = []
        
        for j in range(len(bboxes)):
            if i != j:
                other_bbox = bboxes[j]
                # 计算IoU
                x1 = max(current_bbox[0], other_bbox[0])
                y1 = max(current_bbox[1], other_bbox[1])
                x2 = min(current_bbox[2], other_bbox[2])
                y2 = min(current_bbox[3], other_bbox[3])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
                    area2 = (other_bbox[2] - other_bbox[0]) * (other_bbox[3] - other_bbox[1])
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0
                    overlaps.append(iou)
                else:
                    overlaps.append(0)
        
        # 取最大重叠率
        or_values.append(max(overlaps) if overlaps else 0)
    
    # 动态确定k值
    nb_i = len(bboxes)
    if nb_i <= 5:
        k = nb_i
    else:
        alpha = 0.3
        k = int(alpha * nb_i)
    k = max(1, k)
    
    # 取前k个最大的OR值
    sorted_or = sorted(or_values, reverse=True)
    occlusion_score = np.mean(sorted_or[:k]) if k > 0 else 0.0
    
    return float(occlusion_score)


def calculate_diversity_score(result: DetDataSample) -> float:
    """计算类别多样性系数"""
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()
    
    if len(bboxes) == 0:
        return 0.0
    
    # 计算类别分布
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_proportions = label_counts / len(labels)
    
    # 计算类别因子和熵
    category_factor = np.log(1 + len(unique_labels))
    entropy = np.sum(label_proportions * np.log(label_proportions + 1e-6))
    diversity_score = -(category_factor * entropy)
    
    return float(diversity_score)


def calculate_density_var_score(result: DetDataSample) -> float:
    """计算局部空间密度变异系数"""
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    
    if len(bboxes) <= 1:
        return 0.0
    
    # 计算边界框中心点
    centers = []
    for bbox in bboxes:
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        centers.append([center_x, center_y])
    
    centers = np.array(centers)
    
    # 使用固定参考尺寸进行归一化
    reference_size = 1590.0
    centers_normalized = centers / reference_size
    
    # 使用核密度估计
    kde = KernelDensity(kernel='epanechnikov', bandwidth=0.1).fit(centers_normalized)
    log_densities = kde.score_samples(centers_normalized)
    densities = np.exp(log_densities)
    
    # 归一化密度值
    normalized_densities = (densities - np.min(densities)) / (np.max(densities) - np.min(densities) + 1e-6)
    
    # 计算变异系数
    density_mean = np.mean(normalized_densities) + 1e-6
    density_var_score = np.std(normalized_densities) / density_mean
    
    return float(density_var_score)


# 向后兼容的函数
def recalculate_crown_count_scores(unlabeled_pool_results, unlabeled_pool_results_uncertainty):
    """向后兼容的函数，调用新的SSC重新计算函数"""
    return recalculate_ssc_scores(unlabeled_pool_results, unlabeled_pool_results_uncertainty) 