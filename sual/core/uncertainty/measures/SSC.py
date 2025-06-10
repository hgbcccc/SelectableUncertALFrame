# 3月22日修改版本   3.24续
import numpy as np
from typing import Dict, Optional
from mmdet.structures import DetDataSample
from .utils import _calculate_sor, _get_box_centers
from sklearn.neighbors import KernelDensity

def calculate_ssc(result: DetDataSample, alpha: float = 0.3) -> Dict[str, float]:
    """计算空间结构复杂度(Spatial Structure Complexity, SSC)
    
    综合评估森林冠层的空间分布特性，包含遮挡程度、数量分布、类别多样性、
    尺寸变异和空间密度变异五个维度指标，最终加权得到综合复杂度评分
    
    Args:
        result (DetDataSample): 包含检测结果和元信息的数据结构
            - pred_instances: 预测实例（边界框、标签、置信度）
            - meta: 元信息字典，需包含原始尺寸(ori_shape)和缩放比例(scale_factor)
        alpha (float, optional): 动态top-k选择比例参数（当前版本固定k=10）
    
    Returns:
        Dict[str, float]: 包含各维度指标及综合评分的字典
            - ssc_score: 空间复杂度评分（加权平均）
            - occlusion_score: 遮挡程度指标 [0, +∞)
            - crown_count_score: 数量分布合理性指标 [0, 1]
            - diversity_score: 类别多样性指标 [0, +∞)
            - area_var_score: 尺寸变异指标  [0, +∞]
            - density_var_score: 空间密度变异指标 [0, +∞]
    """
    # --------------------------- 数据准备 ---------------------------
    # 将检测结果从GPU Tensor转换为CPU Numpy数组
    bboxes = result.pred_instances.bboxes.cpu().numpy()  # 边界框坐标数组[N,4]
    labels = result.pred_instances.labels.cpu().numpy()   # 类别标签数组[N]
    
    # 空检测处理：无检测框时返回全零指标
    if len(bboxes) == 0:
        return {
            'ssc_score': 0.0,
            'occlusion_score': 0.0,
            'crown_count_score': 0.0,
            'diversity_score': 0.0,
            'area_var_score': 0.0,
            'density_var_score': 0.0
        }

    # ------------------------ 1. 遮挡系数计算 ------------------------
        # 计算每个边界框的OR（假设_calculate_sor正确实现）
    or_values = [_calculate_sor(bboxes[i], bboxes) for i in range(len(bboxes))]
    # 动态确定 k 值
    nb_i = len(bboxes)
    if nb_i <= 5:
        k = nb_i
    else:
        alpha = 0.3  # 对应文档中的 w_OR
        k = int(alpha * nb_i)
    k = max(1, k)  # 确保 k 至少为 1

    # 取 Top-k 最大的 OR 值平均
    sorted_or = sorted(or_values, reverse=True)
    occlusion_score = np.mean(sorted_or[:k]) if k > 0 else 0.0


    # -------------------- 2. 树冠数量控制系数计算 ---------------------
    # base_score是高斯函数，范围在[0,1]
    # upper_smooth和lower_smooth是sigmoid函数，范围也在[0,1]
    # 三者相乘，结果必然在[0,1]范围内
    crown_count = len(bboxes)  # 当前影像检测框总数
    
    # 使用预设的基准值
    n_batch = 150  # 降低期望检测框数量
    sigma_batch = 50  # 增大标准差容忍度
    
    # 放宽边界阈值
    upper_threshold = n_batch * 2.0  # 从1.5倍放宽到2倍
    lower_threshold = n_batch * 0.3  # 从0.5倍放宽到0.3倍
    
    # 计算基础高斯得分
    normalized_diff = (crown_count - n_batch) / sigma_batch
    base_score = np.exp(-0.5 * normalized_diff ** 2)
    
    # 含过渡和平滑优化
    # 增大过渡区宽度,使得分数变化更平滑
    transition_width = 25  # 从16增加到25
    
    # 调整sigmoid斜率系数,使过渡更平缓
    slope = 8.0  # 从10降低到8
    
    # 上界平滑过渡
    upper_smooth = 1 / (1 + np.exp((crown_count - upper_threshold + transition_width/2)/transition_width*slope))
    
    # 下界平滑过渡  
    lower_smooth = 1 / (1 + np.exp((-crown_count + lower_threshold + transition_width/2)/transition_width*slope))
    
    # 综合得分
    crown_count_score = base_score * upper_smooth * lower_smooth
    
    # 分数截断,提高下限
    # crown_count_score = max(0.00, crown_count_score)

    # --------------------- 3. 类别多样性系数计算 ---------------------
    # 统计类别分布
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_proportions = label_counts / len(labels)  # 各类别占比
    
    # 多样性公式：e_i = -ln(1+|C_i|) × Σ[p(c_k)log p(c_k)] 

    category_factor = np.log(1 + len(unique_labels))          # 正惩罚（类别越多得分越高）
    entropy = np.sum(label_proportions * np.log(label_proportions + 1e-6))  # 正熵（越均匀得分越高）
    diversity_score = category_factor * entropy     # 取值范围 [0, +∞)
    #{'ssc_score': 16.9988, 'occlusion_score': 16.3622, 'crown_count_score': 15.0, 'diversity_score': -0.0001, 'area_var_score': 0.2753, 'density_var_score': 4.8636}
    # 为什么还是出现了负数
    # {'ssc_score': 10.984, 'occlusion_score': 10.401, 'crown_count_score': 15.0, 'diversity_score': -0.0, 'area_var_score': 0.136, 'density_var_score': 4.329}
    # 是因为浮点数精度问题，导致出现了负数，需要转化为整数
    # 废弃的版本
    # ------------------ 4. 边界框面积变异系数计算（修正版）------------------
    # # 从元信息获取原始尺寸和缩放比例
    # ori_h, ori_w = result.ori_shape  # 修改点：直接访问属性 
    # scale = result.scale_factor[0]   # 修改点：直接访问属性  
    # # 计算实际输入尺寸（缩放后的网络输入尺寸）
    # input_h = int(ori_h * scale)  # 如1536*0.333≈512
    # input_w = int(ori_w * scale)
    # max_area = input_h * input_w  # 理论最大边界框面积（全图检测）
    
    # # 计算当前检测框的实际面积
    # areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])  # (x2-x1)*(y2-y1)
    # area_var = np.var(areas) if len(areas) > 0 else 0.0  # 当前面积方差
    
    # # 计算理论最大方差（极端分布：半数最大面积，半数零面积）
    # n = len(areas)
    # if n > 0:
    #     # 构造假设面积数组：前n//2个为最大面积，其余为0
    #     hypothetical_areas = np.concatenate([
    #         np.full(n//2, max_area), 
    #         np.zeros(n - n//2)
    #     ])
    #     sigma_a0 = np.var(hypothetical_areas)  # 理论最大方差
    # else:
    #     sigma_a0 = 1.0  # 防止除零
    
    # # 归一化处理：当前方差/理论最大方差，并限制在[0,1]
    # # 前一个版本问题，论文使用了+1
    # # 计算公式存在缺陷：area_var_score = area_variance / sigma_a0_squared + 1

    # # 应修正为：area_var_score = area_variance / (sigma_a0_squared + 1e-6)

    # # 确保理论最大值σ_a0²=area_mean²时得1
    # area_var_score = np.clip(area_var / (sigma_a0 + 1e-6), 0.0, 1.0)

    # 3月23版本的，考虑到，一批未标注的影像，应该有一个相同的归一化参数，
    # 这里使用的是整个数据结合，最大的图片的尺寸，作为归一化参数

    # ori_h, ori_w = result.ori_shape  # 修改点：直接访问属性 # 原始影像尺寸（如1536x1536）
    # scale = result.scale_factor[0]   # 修改点：直接访问属性 # 缩放比例（如0.333）
    
    # input_size = (int(ori_h * scale), int(ori_w * scale))
    # max_area = input_size[0] * input_size[1]
    max_area = 1590*1590

    # 计算实际面积方差
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    area_var = np.var(areas) if len(areas) > 0 else 0.0

    # 计算理论最大方差
    n = len(areas)
    if n > 0:
        # 构造假设分布：半数最大面积，半数零面积
        n_half = n // 2
        hypothetical_areas = np.concatenate([
            np.full(n_half, max_area),
            np.zeros(n - n_half)
        ])
        sigma_a0 = np.var(hypothetical_areas) #sigma_a0 在论文中指明为归一化参数。
    else:
        sigma_a0 = 1.0  # 空检测时避免除零
    
    area_var_score = (area_var / (sigma_a0 + 1e-6))


    # ------------------ 5. 局部空间密度变异系数计算 ------------------
    centers = _get_box_centers(bboxes)  # 获取边界框中心坐标[N,2]

    if len(centers) > 1:
        # 使用固定值1590进行归一化，
        # 后续需要修改，疑问：这个之应该是整个数据的而不是对这一张影像的，如果是使用这张图片的最大值，会有什么错误还是偏差
        centers_normalized = centers / 1590.0 
        
        # 使用确定的最优参数配置
        kde = KernelDensity(
            kernel='epanechnikov',
            bandwidth=0.1
        ).fit(centers_normalized)
        
        # 计算局部密度
        log_densities = kde.score_samples(centers_normalized)
        densities = np.exp(log_densities)
        
        # 归一化密度值，与实验代码保持一致
        normalized_densities = (densities - np.min(densities)) / (np.max(densities) - np.min(densities) + 1e-6)
        
        # 计算变异系数
        density_mean = np.mean(normalized_densities) + 1e-6  # 防止除零
        # 与实验发现，乘以100得到百位数的变异系数，和遮挡系数量纲不一致，乘10得到十位数的变异系数，与遮挡系数数量纲一致
        # 在后续的比例因子乘10
        density_var_score = np.std(normalized_densities) / density_mean 
        
    else:
        density_var_score = 0.0  # 单个或零检测框时无密度变化


    # ---------------------- 综合评分计算 ----------------------
    # 各指标权重（当前版本等权，可根据需求调整）
    # weights = [1, 0.5, 2, 0.5, 1]  
    # weights = [2, 1, 1, 1, 1]
    weights = [1, 1, 1, 1, 2]    
    
    scores = [
        occlusion_score, 
        crown_count_score, 
        diversity_score,
        area_var_score, 
        density_var_score
    ]
    
    # 加权求和得到最终评分
    ssc_score = float(np.dot(weights, scores))
    
    # 对各个指标进行缩放 ！！方法注释暂时保留！！
    # scale_factors = {
    #     'ssc_score': 1,  # 缩放到10位数
    #     'occlusion_score': 1.0,  # 已经在合适范围
    #     'crown_count_score': 100,  # 缩放到1-2之间
    #     'diversity_score': 100,  # 已经在合适范围
    #     'area_var_score': 1000,  # 缩放到十位数
    #     'density_var_score': 10  # 缩放到十位数
    # }
    # 修改为：不进行量纲的统一，原因，不同的系数。
    # 动态适应性差：固定缩放因子无法适应不同数据集分布特性，当遇到极端样本时会出现指标失真，
    # 修改方法在训练脚本中的normalize_uncertainty_scores函数中，进行的是全局标准化
    scale_factors = {
        'ssc_score': 1,  
        'occlusion_score': 1,  
        'crown_count_score': 1,  
        'diversity_score': 1, 
        'area_var_score': 1,  
        'density_var_score': 1  
    }
    # 取小数点后4位
    # return {
    #     'ssc_score': round(ssc_score * scale_factors['ssc_score'], 2),
    #     'occlusion_score': round(float(occlusion_score) * scale_factors['occlusion_score'], 2),
    #     'crown_count_score': round(float(crown_count_score) * scale_factors['crown_count_score'], 2),
    #     'diversity_score': round(float(diversity_score) * scale_factors['diversity_score'], 2),
    #     'area_var_score': round(float(area_var_score) * scale_factors['area_var_score'], 2),
    #     'density_var_score': round(float(density_var_score) * scale_factors['density_var_score'], 2)
    # }
    # 在SSC.py中修改返回部分
    return {
        'ssc_score': int(ssc_score * scale_factors['ssc_score'] * 100) / 100,
        'occlusion_score': int(float(occlusion_score) * scale_factors['occlusion_score'] * 100) / 100,
        'crown_count_score': int(float(crown_count_score) * scale_factors['crown_count_score'] * 100) / 100,
        'diversity_score': int(float(-diversity_score) * scale_factors['diversity_score'] * 100) / 100,  # 保留负号
        'area_var_score': int(float(area_var_score) * scale_factors['area_var_score'] * 100) / 100,
        'density_var_score': int(float(density_var_score) * scale_factors['density_var_score'] * 100) / 100
    }