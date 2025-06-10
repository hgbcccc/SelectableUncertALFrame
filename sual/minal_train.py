
# import sys
# # sys.path.append('E:\\sual')s
# import sys
# import locale
# sys.stdout.reconfigure(encoding='utf-8')

###########################################输出###########################################

# '25f40233-1375-46e9-b31b-ab5b5ae1adc8.jpg': {'result': <DetDataSample(

#     META INFORMATION
#     scale_factor: (0.3333333333333333, 0.3333333333333333)
#     img_path: None
#     ori_shape: (1536, 1536)
#     batch_input_shape: (512, 512)
#     img_shape: (512, 512)
#     pad_shape: (512, 512)
#     img_id: 0

#     DATA FIELDS
#     ignored_instances: <InstanceData(
        
#             META INFORMATION
        
#             DATA FIELDS
#             labels: tensor([], device='cuda:0', dtype=torch.int64)
#             bboxes: tensor([], device='cuda:0', size=(0, 4))
#         ) at 0x78ccb449dd30>
#     gt_instances: <InstanceData(
        
#             META INFORMATION
        
#             DATA FIELDS
#             labels: tensor([], device='cuda:0', dtype=torch.int64)
#             bboxes: tensor([], device='cuda:0', size=(0, 4))
#         ) at 0x78ccb449d640>
#     pred_instances: <InstanceData(
        
#             META INFORMATION
        
#             DATA FIELDS
#             scores: tensor([0.3784, 0.3743, 0.3621, 0.3619, 0.3578, 0.3568, 0.3550, 0.3537, 0.3513,
#                         0.3490, 0.3471, 0.3457, 0.3408, 0.3370, 0.3361, 0.3361, 0.3353, 0.3339,
#                         0.3337, 0.3324, 0.3317, 0.3280, 0.3249, 0.3232, 0.3203, 0.3194, 0.3192,
#                         0.3187, 0.3181, 0.3180, 0.3170, 0.3161, 0.3157, 0.3154, 0.3149, 0.3139,
#                         0.3122, 0.3115, 0.3115, 0.3115, 0.3055, 0.3034, 0.2996, 0.2995, 0.2976,
#                         0.2975, 0.2972, 0.2969, 0.2967, 0.2959, 0.2958, 0.2955, 0.2954, 0.2953,
#                         0.2932, 0.2919, 0.2908, 0.2899, 0.2895, 0.2889, 0.2887, 0.2846, 0.2843,
#                         0.2830, 0.2828, 0.2813, 0.2807, 0.2802, 0.2799, 0.2794, 0.2794, 0.2788,
#                         0.2783, 0.2771, 0.2770, 0.2765, 0.2753, 0.2744, 0.2743, 0.2732, 0.2727,
#                         0.2725, 0.2713, 0.2703, 0.2692, 0.2685, 0.2684, 0.2674, 0.2672, 0.2669,
#                         0.2661, 0.2656, 0.2649, 0.2647, 0.2645, 0.2640, 0.2628, 0.2620, 0.2609,
#                         0.2604, 0.2592, 0.2585, 0.2579, 0.2578, 0.2576, 0.2571, 0.2565, 0.2556,
#                         0.2550, 0.2548, 0.2537, 0.2520, 0.2518, 0.2514, 0.2505, 0.2499, 0.2496,
#                         0.2493, 0.2488, 0.2481, 0.2480, 0.2477, 0.2464, 0.2461, 0.2460, 0.2458,
#                         0.2454, 0.2454, 0.2452, 0.2451, 0.2449, 0.2445, 0.2443, 0.2442, 0.2422,
#                         0.2417, 0.2414, 0.2396, 0.2384, 0.2382, 0.2377, 0.2374, 0.2372, 0.2362,
#                         0.2360, 0.2356, 0.2354, 0.2348, 0.2347, 0.2346, 0.2341, 0.2335, 0.2331,
#                         0.2328, 0.2325, 0.2324, 0.2320, 0.2303, 0.2299, 0.2296, 0.2275, 0.2264,
#                         0.2264, 0.2258, 0.2256, 0.2255, 0.2249, 0.2247, 0.2246, 0.2245, 0.2242,
#                         0.2242, 0.2241, 0.2238, 0.2235, 0.2235, 0.2228, 0.2222, 0.2207, 0.2203,
#                         0.2196, 0.2193, 0.2191, 0.2187, 0.2187, 0.2184, 0.2183, 0.2182, 0.2179,
#                         0.2176, 0.2174, 0.2173, 0.2172, 0.2162, 0.2161, 0.2158, 0.2151, 0.2147,
#                         0.2144, 0.2144, 0.2134, 0.2133, 0.2132, 0.2127, 0.2119, 0.2119, 0.2118,
#                         0.2117, 0.2114, 0.2105, 0.2104, 0.2102, 0.2100, 0.2090, 0.2088, 0.2085,
#                         0.2078, 0.2075, 0.2074, 0.2065, 0.2064, 0.2063, 0.2063, 0.2061, 0.2053,
#                         0.2049, 0.2036, 0.2035, 0.2033, 0.2032, 0.2021, 0.2021, 0.2017, 0.2015,
#                         0.2010, 0.2010, 0.2008, 0.2005, 0.2002, 0.2001, 0.1996, 0.1991, 0.1988,
#                         0.1986, 0.1985, 0.1984, 0.1984, 0.1983, 0.1978, 0.1977, 0.1976, 0.1973,
#                         0.1966, 0.1961, 0.1947, 0.1945, 0.1944, 0.1941, 0.1939, 0.1937, 0.1936,
#                         0.1935, 0.1934, 0.1930, 0.1927, 0.1914, 0.1912, 0.1909, 0.1907, 0.1904,
#                         0.1903, 0.1902, 0.1896, 0.1894, 0.1893, 0.1892, 0.1886, 0.1882, 0.1878,
#                         0.1875, 0.1875, 0.1865, 0.1864, 0.1861, 0.1860, 0.1852, 0.1850, 0.1849,
#                         0.1842, 0.1841, 0.1840, 0.1839, 0.1839, 0.1837, 0.1835, 0.1834, 0.1832,
#                         0.1831, 0.1830, 0.1820], device='cuda:0')
#             labels: tensor([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#                         4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#                         4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#                         4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#                         4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#                         4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#                         4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#                         4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#                         4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4,
#                         4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 4, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4,
#                         4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 4, 3, 4,
#                         4, 4, 4, 3, 4, 4, 4, 3, 3, 4, 3, 4, 3, 3, 3, 4, 3, 3, 4, 4, 3, 4, 3, 3,
#                         3, 4, 3, 4, 3, 3, 3, 4, 4, 3, 3, 4], device='cuda:0')
#             bboxes: tensor([[1145.7115,  781.0701, 1313.5216, 1017.6187],
#                         [1171.6738,  707.8335, 1336.8342,  944.3883],
#                         [1200.6938,  749.5586, 1364.7812,  993.6741],
#                         ...,
#                         [1133.9071,  403.3912, 1215.4822,  539.5458],
#                         [1121.6398,  385.4045, 1201.4153,  527.2668],
#                         [ 737.2630,  215.9899,  909.0403,  453.9951]], device='cuda:0')
#         ) at 0x78ccb449da90>
# ) at 0x78ccb449de50>, 
# 'vis_path': 'work_dirs/al_ssc/round_1/teacher_outputs/20250318_175214/visualize/25f40233-1375-46e9-b31b-ab5b5ae1adc8_vis.jpg',
# 'uncertainty': {'ssc_score': 24.451695791348588, 'occlusion_score': 21.81858649915016, 'crown_count_score': 3.410310592094398e-11, 'diversity_score': 0.3806838101318505, 'area_var_score': 1.5647551043417407, 'density_var_score': 0.6876703776907318}}}


import argparse
import numpy as np
from pathlib import Path
import json
import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from sual.inference.simdetector import SimDetectionInference
from sual.core.datasets import ActiveCocoDataset
import os.path as osp
import re
from mmengine.logging import MMLogger
from sual.core.hooks import ActiveLearningEvalHook
from datetime import datetime
import pandas as pd
from scipy.stats import wasserstein_distance
from typing import Dict, List, Optional, Tuple


"""
主动学习训练脚本

该脚本实现了一个主动学习的训练流程，主要包括以下步骤：

1. **推理训练集**：
   使用目标检测模型对标注的训练集进行推理，得到每张图片的检测结果和不确定性指标。

   示例数据结构：
   result_un = {
       'image1.jpg': {
           'result': DetDataSample对象,  # 检测结果
           'uncertainty': {
               'occlusion_score': 28.20,  # 遮挡分数
               'crown_count_score': 3.41e-11,  # 树冠数量控制分数
               'diversity_score': 0.038,  # 多样性分数
               'area_var_score': 2.0,  # 面积变异分数
               'density_var_score': 0.91,  # 密度变异分数
               'ssc_score': 10.0  # 原始空间结构复杂度分数
           }
       },
       'image2.jpg': {
           'result': DetDataSample对象,
           'uncertainty': {
               'occlusion_score': 30.50,
               'crown_count_score': 1.20e-10,
               'diversity_score': 0.045,
               'area_var_score': 1.8,
               'density_var_score': 0.85,
               'ssc_score': 12.0
           }
       }
   }

2. **计算不确定性**：
   对推理结果调用`compute_uncertainty`方法，计算每张图片的多个不确定性指标。

3. **使用WassersteinBalancedScorer**：
   创建`WassersteinBalancedScorer`实例，计算每个样本的平衡得分，并更新`ssc_score`。

   示例数据结构：
   processed_results = {
       'image1.jpg': {
           'result': DetDataSample对象,
           'vis_path': 'path/to/visualization.jpg',
           'uncertainty': {
               'occlusion_score': 28.20,
               'crown_count_score': 3.41e-11,
               'diversity_score': 0.038,
               'area_var_score': 2.0,
               'density_var_score': 0.91,
               'ssc_score': 15.0,  # 更新后的分数
               'wasserstein_balanced_score': 20.0,  # 新的平衡得分
               'w_score': 5.0,  # 线性组合得分
               'mapd_score': 0.1,  # MAPD得分
               'perturbation_score': 1.5,  # 扰动得分
               'feature_wasserstein_distances': {...}  # 各特征的Wasserstein距离
           }
       },
       'image2.jpg': {...}
   }

4. **选择样本**：
   使用`dataset.select_samples`方法，根据计算出的平衡得分选择样本。

5. **更新数据集**：
   将选中的样本更新到数据集中。

示例输出：
    result_un 中的 ssc_score 平均值: 11.0
    processed_results 中的 ssc_score 平均值: 16.5
    选择完成，选中样本数量: 5
    数据集更新成功

通过这种方式，脚本能够有效地选择未标注样本，增强模型的泛化能力。
"""

class WassersteinBalancedScorer:
    """基于Wasserstein距离的平衡评分器"""
    
    def __init__(self,
                 alpha: float = 0.5,    # 线性组合权重
                 beta: float = 0.3,     # 紧凑性权重
                 gamma: float = 0.2,    # 分布扰动权重
                 mapd_threshold: float = 0.25,  # MAPD阈值(25%分位数)
                 feature_names: List[str] = [
                     'occlusion_score', 'crown_count_score', 'diversity_score',
                     'area_var_score', 'density_var_score'
                 ]):
        """
        Args:
            alpha: 线性组合项权重
            beta: 紧凑性项权重
            gamma: 分布扰动项权重
            mapd_threshold: MAPD阈值
            feature_names: 特征名称列表
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mapd_threshold = mapd_threshold
        self.feature_names = feature_names
        
        # 用于存储训练集统计信息
        self.train_stats = {
            'mean': None,
            'std': None,
            'distributions': None
        }

    def _standardize_features(self, features: Dict[str, float]) -> np.ndarray:
        """标准化特征
        
        Args:
            features: 特征字典
            
        Returns:
            标准化后的特征数组
        """
        # 提取特征值
        feature_values = np.array([features[name] for name in self.feature_names])
        
        # 如果没有训练集统计信息，直接返回原始值
        if self.train_stats['mean'] is None or self.train_stats['std'] is None:
            return feature_values
            
        # 标准化
        return (feature_values - self.train_stats['mean']) / (self.train_stats['std'] + 1e-6)

    def _calculate_mapd(self, standardized_features: np.ndarray) -> float:
        """计算平均绝对特征差距(MAPD)"""
        n = len(standardized_features)
        diffs = []
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(abs(standardized_features[i] - standardized_features[j]))
        return np.mean(diffs) if diffs else 0.0
    
    def _calculate_wasserstein_distances(self, 
                                       features: Dict[str, float]) -> Dict[str, float]:
        """计算每个特征的Wasserstein距离"""
        distances = {}
        standardized = self._standardize_features(features)
        
        for i, name in enumerate(self.feature_names):
            # 如果没有训练集分布信息，返回0
            if self.train_stats['distributions'] is None:
                distances[name] = 0.0
                continue
                
            # 模拟将新样本加入训练集
            new_distribution = np.append(self.train_stats['distributions'][name], 
                                       standardized[i])
            # 计算Wasserstein距离
            distances[name] = wasserstein_distance(
                self.train_stats['distributions'][name],
                new_distribution
            )
        return distances
    
    # def compute_balanced_score(self, uncertainty_metrics: Dict[str, float]) -> Dict:
    #     """计算平衡后的得分"""
    #     # 1. 提取特征值
    #     features = {name: uncertainty_metrics[name] 
    #                for name in self.feature_names}
        
    #     # 2. 标准化特征
    #     standardized_features = self._standardize_features(features)
        
    #     # 3. 计算线性组合得分(w)
    #     w_score = np.sum(standardized_features)
        
    #     # 4. 计算MAPD
    #     mapd = self._calculate_mapd(standardized_features)
    #     mapd_score = 1 - (mapd / (self.mapd_threshold + 1e-6))
        
    #     # 5. 计算Wasserstein距离
    #     w_distances = self._calculate_wasserstein_distances(features)
    #     perturbation_score = np.sum(list(w_distances.values()))
        
    #     # 6. 计算综合得分
    #     final_score = (
    #         self.alpha * (w_score / (np.max(np.abs(w_score)) + 1e-6)) +
    #         self.beta * mapd_score +
    #         self.gamma * (perturbation_score / (np.max(perturbation_score) + 1e-6))
    #     )
        
    #     # 构建返回结果
    #     result = uncertainty_metrics.copy()
    #     result.update({
    #         'wasserstein_balanced_score': final_score * 100,  # 转换到相似范围
    #         'w_score': w_score,
    #         'mapd_score': mapd_score,
    #         'perturbation_score': perturbation_score,
    #         'feature_wasserstein_distances': w_distances
    #     })
        
    #     return result
    def compute_balanced_score(self, uncertainty_metrics: Dict[str, float]) -> Dict:
        """计算平衡后的得分"""
        # # 1. 提取特征值
        # features = {name: uncertainty_metrics[name] for name in self.feature_names}
        
        # # 打印特征值
        # print(f"特征值: {features}")
        # print(f"标准化特征: {standardized_features}")

        # # 2. 标准化特征
        # standardized_features = self._standardize_features(features)

        # # 打印标准化后的特征
        # print(f"标准化特征: {standardized_features}")

        # # 3. 计算线性组合得分(w)
        # w_score = np.sum(standardized_features)
        # print(f"w_score: {w_score}")
        try:
            print("调用 compute_balanced_score 方法")
            features = {name: uncertainty_metrics[name] for name in self.feature_names}
            standardized_features = self._standardize_features(features)

            print(f"特征值: {features}")
            print(f"标准化特征: {standardized_features}")

            w_score = np.sum(standardized_features)
            print(f"w_score: {w_score}")

        except Exception as e:
            print(f"计算平衡得分时出错: {e}")
        # 4. 计算MAPD
        mapd = self._calculate_mapd(standardized_features)
        mapd_score = 1 - (mapd / (self.mapd_threshold + 1e-6))

        # 5. 计算Wasserstein距离
        w_distances = self._calculate_wasserstein_distances(features)
        perturbation_score = np.sum(list(w_distances.values()))

        # 6. 计算综合得分
        final_score = (
            self.alpha * (w_score / (np.max(np.abs(w_score)) + 1e-6)) +
            self.beta * mapd_score +
            self.gamma * (perturbation_score / (np.max(perturbation_score) + 1e-6))
        )

        # 构建返回结果
        result = uncertainty_metrics.copy()
        result.update({
            'wasserstein_balanced_score': final_score * 100,  # 转换到相似范围
            'w_score': w_score,
            'mapd_score': mapd_score,
            'perturbation_score': perturbation_score,
            'feature_wasserstein_distances': w_distances
        })

        return result
    
    def rank_samples(self, 
                    samples_metrics: Dict[str, Dict],
                    top_m: int = 50) -> List[str]:
        """对样本进行排序"""
        # 1. 计算所有样本的平衡得分
        scored_samples = []
        for sample_id, metrics in samples_metrics.items():
            if 'uncertainty' in metrics:
                balanced_metrics = self.compute_balanced_score(
                    metrics['uncertainty'])
                scored_samples.append(
                    (sample_id, balanced_metrics['wasserstein_balanced_score']))
        
        # 2. 按得分降序排序并选择top_m
        scored_samples.sort(key=lambda x: x[1], reverse=True)
        return [sample_id for sample_id, _ in scored_samples[:top_m]]


def find_best_checkpoint(work_dir: Path, logger: Optional[MMLogger] = None) -> Optional[str]:
    """查找最佳检查点
    
    策略：
    1. 从日志中查找最佳检查点信息
    2. 在工作目录中查找所有检查点
    3. 按照不同类型的检查点进行优先级排序
    """
    work_dir = Path(work_dir)
    
    # 1. 从日志文件中查找最佳检查点
    log_file = work_dir / 'run.log'
    best_ckpt = None
    
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_content = f.read()
            # 使用正则表达式匹配最后一次保存的最佳检查点
            matches = re.finditer(
                r'The best checkpoint .+ is saved to (.+\.pth)',
                log_content
            )
            # 获取最后一个匹配结果
            best_ckpt_matches = list(matches)
            if best_ckpt_matches:
                best_ckpt = best_ckpt_matches[-1].group(1)
                best_ckpt = work_dir / best_ckpt
                if best_ckpt.exists():
                    if logger:
                        logger.info(f'从日志中找到最佳检查点: {best_ckpt}')
                    return str(best_ckpt)
    
    # 2. 在工作目录中查找所有检查点
    def get_checkpoint_priority(ckpt_path: Path) -> int:
        """定义检查点的优先级"""
        name = ckpt_path.name
        if 'best' in name and 'bbox_mAP' in name:
            return 4  # 最高优先级：性能最好的检查点
        if 'best' in name:
            return 3  # 其他最佳检查点
        if 'epoch' in name:
            return 2  # epoch 检查点
        return 1  # 其他检查点
    
    # 递归查找所有 .pth 文件
    checkpoints: List[Path] = []
    for ext in ['.pth', '.pt', '.ckpt']:  # 支持多种扩展名
        checkpoints.extend(work_dir.rglob(f'*{ext}'))
    
    if not checkpoints:
        if logger:
            logger.warning(f'在 {work_dir} 中未找到任何检查点')
        return None
    
    # 按优先级和修改时间排序
    checkpoints.sort(
        key=lambda x: (
            get_checkpoint_priority(x),  # 首先按优先级
            x.stat().st_mtime  # 然后按修改时间
        ),
        reverse=True
    )
    
    best_ckpt = str(checkpoints[0])
    if logger:
        logger.info(f'找到最佳检查点: {best_ckpt}')
    return best_ckpt


def parse_args():
    parser = argparse.ArgumentParser(description='主动学习训练')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--work-dir', help='工作目录')
    parser.add_argument('--cfg-options',nargs='+',action=DictAction,help='覆盖配置文件中的选项')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    
    # 加载配置
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    logger = MMLogger.get_current_instance()
    
    # 打印数据集路径信息
    print(f"数据根目录: {cfg.data_root}")
    print(f"训练集图片目录: {cfg.train_dataloader.dataset.data_prefix['img']}")
    print(f"训练集标注文件: {cfg.train_dataloader.dataset.ann_file}")
    
    # 检查文件是否存在
    img_dir = cfg.train_dataloader.dataset.data_prefix['img']
    if not osp.exists(img_dir):
        print(f"警告: 图片目录不存在: {img_dir}")
    
    ann_file = cfg.train_dataloader.dataset.ann_file
    if not osp.exists(ann_file):
        print(f"警告: 标注文件不存在: {ann_file}")
    
    # 设置工作目录
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = Path('./work_dirs') / Path(args.config).stem
        
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取主动学习配置
    al_cfg = cfg.active_learning
    
    # 初始化性能跟踪
    performance_history = {
        'round': [],
        'labeled_ratio': [],        # 标注比例
        'labeled_images': [],       # 已标注图片数
        'unlabeled_images': [],     # 未标注图片数
        'total_images': [],         # 总图片数
        'labeled_annotations': [],   # 已标注标注框数量
        'val_bbox_mAP': [],        # 验证集性能
        'val_bbox_mAP_50': [],
        'val_bbox_mAP_75': [],
        'timestamp': []             # 时间戳
    }
    
    # 主动学习循环
    for active_learning_round in range(1, al_cfg.max_iterations + 1):
        print(f"\n开始第 {active_learning_round}/{al_cfg.max_iterations} 轮主动学习...")

        
        # 创建当前迭代的工作目录
        iter_work_dir = work_dir / f"round_{active_learning_round}"
        iter_work_dir.mkdir(exist_ok=True)
        
        # 更新配置中的工作目录
        cfg.work_dir = str(iter_work_dir)
        
        # 如果不是第一轮，加载上一轮的最佳模型
        if active_learning_round > 1:
            prev_iter_dir = work_dir / f"round_{active_learning_round - 1}"
            prev_ckpt = find_best_checkpoint(prev_iter_dir, logger)
            if prev_ckpt:
                logger.info(f"加载上一轮检查点: {prev_ckpt}")
                cfg.load_from = prev_ckpt
            else:
                logger.warning(f"未找到上一轮检查点")
        
        # 1. 训练学生模型
        runner = Runner.from_cfg(cfg)
        runner.train()
        
        # 2. 评估模型性能
        eval_results = {}
        try:
            # 验证集评估
            if hasattr(cfg, 'val_dataloader') and hasattr(cfg, 'val_evaluator'):
                val_results = runner.val()
                # 打印原始结果以便调试
                # logger.info(f"验证集原始结果: {val_results}")  # 已经存在的输出，是由mmdet mmengine给出的验证机评估结果，不需要在进行打印
                # 确保获取到正确的指标
                if isinstance(val_results, dict):
                    val_metrics = val_results.get('coco/bbox_mAP', 0.0)
                    val_metrics_50 = val_results.get('coco/bbox_mAP_50', 0.0)
                    val_metrics_75 = val_results.get('coco/bbox_mAP_75', 0.0)
                    val_metrics_95 = val_results.get('coco/bbox_mAP_95', 0.0)
                else:
                    val_metrics = val_metrics_50 = val_metrics_75 = 0.0
                eval_results['val'] = {
                    'bbox_mAP': val_metrics,
                    'bbox_mAP_50': val_metrics_50,
                    'bbox_mAP_75': val_metrics_75,
                    'bbox_mAP_95': val_metrics_95
                }
                # 使得输出结果更简洁  
                formatted_result = ", ".join([f"{key}: {value}" for key, value in eval_results['val'].items()])
                logger.info(f"验证集评估结果: {formatted_result}")   # 验证集评估结果: bbox_mAP: 0.001, bbox_mAP_50: 0.008, bbox_mAP_75: 0.0  # 新添加  bbox_mAP_95
        except Exception as e:
            logger.warning(f"评估过程出错: {e}")
            eval_results = {'val': {}}
        
        # 3. 使用训练好的模型进行推理   # 后续应该需要直接使用是训练集中GT来计算，不是使用模型推理的结果
        latest_ckpt = find_best_checkpoint(iter_work_dir, logger)
        if not latest_ckpt:
            raise FileNotFoundError(f"在 {iter_work_dir} 中未找到有效的检查点文件")
                 
        # 3.1 对训练集进行推理和不确定性计算
        logger.info("开始对训练集进行推理...")
        train_teacher = SimDetectionInference(
            config_file=args.config,
            checkpoint_file=latest_ckpt,
            output_dir=str(iter_work_dir / 'train_inference'),
            enable_uncertainty=True,
            uncertainty_methods=al_cfg.inference_options.uncertainty_methods
        )
        
        # 对训练集进行推理
        train_results = train_teacher.inference(
            str(Path(al_cfg.data_root) / 'images_labeled_train'),
            save_vis=False  # 训练集不需要保存可视化结果
        )
        # print("训练集的推理结果")
        # print(train_results)
        # 计算训练集的不确定性
        train_uncertainty = train_teacher.compute_uncertainty(
            train_results,
            score_thr=al_cfg.inference_options['score_thr']
        )
        
        # 收集训练集特征统计信息
        train_features = {
            'occlusion_score': [],
            'crown_count_score': [],
            'diversity_score': [],
            'area_var_score': [],
            'density_var_score': []
        }
        
        detection_counts = []  # 用于记录每张图片的检测框数量
        
        # 收集特征值
        for img_name, info in train_uncertainty.items():
            if 'uncertainty' in info:
                uncertainty = info['uncertainty']
                for feature_name in train_features.keys():
                    if feature_name in uncertainty:
                        train_features[feature_name].append(uncertainty[feature_name])
                
                # 记录检测框数量
                if 'result' in info:
                    detection_counts.append(
                        len(info['result'].pred_instances.scores)
                    )
        
        # 计算统计量
        train_stats = {
            'features': {
                name: {
                    'mean': float(np.mean(values)) if values else 0.0,
                    'std': float(np.std(values)) if values else 0.0,
                    'min': float(np.min(values)) if values else 0.0,
                    'max': float(np.max(values)) if values else 0.0
                }
                for name, values in train_features.items()
            },
            'batch_statistics': {
                'mean': float(np.mean(detection_counts)) if detection_counts else 0.0,
                'std': float(np.std(detection_counts)) if detection_counts else 0.0,
                'min': int(np.min(detection_counts)) if detection_counts else 0,
                'max': int(np.max(detection_counts)) if detection_counts else 0
            }
        }
        
        # print(train_stats)
        # {'features': {
        #     'occlusion_score': {'mean': 28.205266074314018, 'std': 9.142150468696272, 
        #                          'min': 11.6704776339679, 'max': 64.8474034638837}, 
        #     'crown_count_score': {'mean': 3.410310592094397e-11, 'std': 6.462348535570529e-27, 
        #                           'min': 3.410310592094398e-11, 'max': 3.410310592094398e-11}, 
        #     'diversity_score': {'mean': 0.03803241345202458, 'std': 0.0644748022687212, 
        #                         'min': -6.931468339295632e-07, 'max': 0.5010756428660651}, 
        #     'area_var_score': {'mean': 2.0, 'std': 8.688003089900535e-17, 
        #                        'min': 1.9999999999999996, 'max': 2.0}, 
        #     'density_var_score': {'mean': 0.9169257398018467, 'std': 0.13595889834419558, 
        #                           'min': 0.48201570980447334, 'max': 1.0}}, 
        #     'batch_statistics': {'mean': 300.0, 'std': 0.0, 'min': 300, 'max': 300}
        # }
        # 保存训练集统计信息
        stats_path = iter_work_dir / 'train_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(train_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练集统计信息已保存至: {stats_path}")

        # 4. 推理未标注数据 
        logger.info("开始推理未标注数据...")
        teacher = SimDetectionInference(
            config_file=args.config,
            batch_size=4,
            checkpoint_file=latest_ckpt,
            output_dir=str(iter_work_dir / 'teacher_outputs'),
            enable_uncertainty=True,
            uncertainty_methods=al_cfg.inference_options.uncertainty_methods
        )
        logger.info(f"未标注池中随机采样：{al_cfg.inference_options.sample_size}张")

        # 推理未标注数据
        results = teacher.inference(
            str(Path(al_cfg.data_root) / 'images_unlabeled'),
            sample_size= al_cfg.inference_options.sample_size
 
        )

        result_un = teacher.compute_uncertainty(
            results,
            score_thr=al_cfg.inference_options['score_thr']
        )
        # print(result_un) 

        # 计算并打印 result_un 中的 ssc_score 平均值
        ssc_scores_result_un = []
        for img_name, info in result_un.items():
            if 'uncertainty' in info and 'ssc_score' in info['uncertainty']:
                ssc_scores_result_un.append(info['uncertainty']['ssc_score'])

        if ssc_scores_result_un:
            avg_ssc_score_result_un = np.mean(ssc_scores_result_un)
            print(f"result_un 中的 ssc_score 平均值: {avg_ssc_score_result_un:.4f}")
        else:
            print("result_un 中没有有效的 ssc_score")

        # 定义主动学习数据集类
        dataset = ActiveCocoDataset(
            data_root=al_cfg.data_root,
            ann_file=cfg.train_dataloader.dataset.ann_file,
            data_prefix=cfg.train_dataloader.dataset.data_prefix
        )
        #定义Wasserstein来计算未标注池和标注训练池的分布差距
        wasserstein_scorer = WassersteinBalancedScorer(
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            mapd_threshold=0.25
        )
        # 构建与dataset.select_samples相同格式的输入 这样可能不经过activate_dataset.py的select_samples的排序并且选择样本，
        # 直接使用WassersteinBalancedScorer的排序并选择
        processed_results = {}
        for img_name, info in result_un.items():
            if 'uncertainty' in info:
                # 计算Wasserstein平衡得分
                balanced_metrics = wasserstein_scorer.compute_balanced_score(info['uncertainty'])
                # 保持与原始uncertainty相同的结构，但更新ssc_score
                info['uncertainty']['ssc_score'] = balanced_metrics['wasserstein_balanced_score']
                processed_results[img_name] = info

        
        # 计算未标注数据的不确定性
        # 使用dataset的select_samples方法进行选择
        # 5. 选择新样本
        logger.info("开始选择新样本...")
        logger.info(f"sample_selection 参数: {al_cfg.sample_selection}")
        # print(f"processed_results: {processed_results}")
        # print(processed_results)

        # 计算并打印 processed_results 中的 ssc_score 平均值
        ssc_scores_processed_results = []
        for img_name, info in processed_results.items():
            if 'uncertainty' in info and 'ssc_score' in info['uncertainty']:
                ssc_scores_processed_results.append(info['uncertainty']['ssc_score'])

        if ssc_scores_processed_results:
            avg_ssc_score_processed_results = np.mean(ssc_scores_processed_results)
            print(f"processed_results 中的 ssc_score 平均值: {avg_ssc_score_processed_results:.4f}")
        else:
            print("processed_results 中没有有效的 ssc_score")

        selected_samples = dataset.select_samples(
            results=processed_results,
            **al_cfg.sample_selection
        )
        logger.info(f"选择完成，选中样本数量: {len(selected_samples)}")
        logger.info("开始更新数据集...")
        
        # 6. 更新数据集
        success = dataset.update_dataset(selected_samples)
        if not success:
            logger.error("数据集更新失败")
            raise RuntimeError("数据集更新失败")
        logger.info("数据集更新成功")
        
        # 7. 更新性能历史
        current_stats = dataset.get_dataset_stats()
        performance_history['round'].append(active_learning_round)
        performance_history['labeled_ratio'].append(current_stats['labeled_ratio'])
        performance_history['labeled_images'].append(current_stats['labeled_images'])
        performance_history['unlabeled_images'].append(current_stats['unlabeled_images'])
        performance_history['total_images'].append(current_stats['total_images'])
        performance_history['labeled_annotations'].append(current_stats['labeled_annotations'])
        
        # 添加验证集性能
        val_results = eval_results.get('val', {})
        performance_history['val_bbox_mAP'].append(val_results.get('bbox_mAP', 0.0))
        performance_history['val_bbox_mAP_50'].append(val_results.get('bbox_mAP_50', 0.0))
        performance_history['val_bbox_mAP_75'].append(val_results.get('bbox_mAP_75', 0.0))
        
        # 添加时间戳
        performance_history['timestamp'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # 8. 保存统计信息
        stats_info = {
            'iteration': active_learning_round,
            'selected_samples': selected_samples,
            'dataset_stats': current_stats,
            'evaluation_results': eval_results
        }
            
        with open(iter_work_dir / 'stats.json', 'w') as f:
            json.dump(stats_info, f, indent=2)
        
        # 9. 保存性能历史到CSV
        df = pd.DataFrame(performance_history)
        df.to_csv(work_dir / 'performance_history.csv', index=False)
        
        # 10. 打印当前轮次的详细信息
        print(f"\n第 {active_learning_round} 轮统计信息:")
        print(f"数据集统计:")
        print(f"  - 已标注图片数: {current_stats['labeled_images']}")
        print(f"  - 未标注图片数: {current_stats['unlabeled_images']}")
        print(f"  - 总图片数: {current_stats['total_images']}")
        print(f"  - 标注比例: {current_stats['labeled_ratio']:.2%}")
        print(f"  - 已标注框数量: {current_stats['labeled_annotations']}")
        
        if val_results:
            print(f"验证集性能:")
            print(f"  - bbox_mAP: {val_results.get('bbox_mAP', 0.0):.4f}")
            print(f"  - bbox_mAP_50: {val_results.get('bbox_mAP_50', 0.0):.4f}")
            print(f"  - bbox_mAP_75: {val_results.get('bbox_mAP_75', 0.0):.4f}")
        
        # 清理 GPU 内存
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
