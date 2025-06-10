from typing import Dict, List, Tuple, Set
import numpy as np
from scipy.stats import wasserstein_distance

class WassersteinSampleSelector:
    """基于 Wasserstein 距离的主动学习样本选择器。
    
    通过计算候选样本与训练集特征分布的 Wasserstein 距离，结合 SSC 分数和特征平衡性，
    选择对模型训练最有价值的样本。适用于需要分布匹配的主动学习场景。

    Attributes:
        train_stats (Dict): 训练集的统计信息，包含各特征的分布。
        features (List[str]): 用于计算距离的特征列表，默认包括:
            - occlusion_score
            - crown_count_score
            - diversity_score
            - area_var_score
            - density_var_score

    Example:
        >>> selector = WassersteinSampleSelector(train_stats)
        >>> selected_samples = selector.select_samples(unlabeled_stats, select_num=10)
        >>> report = selector.get_selection_report(selected_samples, unlabeled_stats)
    """

    def __init__(self, train_stats: Dict):
        """初始化选择器。

        Args:
            train_stats: 训练集统计信息，需包含:
                - features: 各特征的分布（如 `occlusion_score.distribution`）
                - 其他必要的统计量（如 mean/std）
        """
        self.train_stats = train_stats
        self.features = [
            'occlusion_score',
            'crown_count_score',
            'diversity_score',
            'area_var_score',
            'density_var_score'
        ]

    def _calculate_wasserstein(self, feature: str, new_value: float) -> float:
        """计算添加新样本后，该特征与训练集分布的 Wasserstein 距离。

        Args:
            feature: 特征名称（如 'occlusion_score'）
            new_value: 新样本的特征值

        Returns:
            float: Wasserstein 距离值（值越大表示分布差异越大）

        Note:
            使用 `scipy.stats.wasserstein_distance` 实现。
        """
        original_dist = np.array(self.train_stats['features'][feature]['distribution'])
        new_dist = np.append(original_dist, new_value)
        return wasserstein_distance(original_dist, new_dist)

    def _calculate_balance_score(self, feature_values: List[float]) -> float:
        """计算多个特征之间的平衡性得分（基于变异系数）CV 越小 → 平衡性越好 → 得分越高（1 / (1 + cv) 确保 CV 越小，得分越高）。。

        Args:
            feature_values: 各特征的值列表，顺序与 `self.features` 一致

        Returns:
            float: 平衡性得分（范围 0~1，值越大表示特征间越平衡）

        Formula:
            score = 1 / (1 + CV), 其中 CV = std(feature_values) / mean(feature_values)
        """
        cv = np.std(feature_values) / (np.mean(feature_values) + 1e-6)
        
        return 1 / (1 + cv)
    

    def _calculate_difference_score(self, sample_values: List[float]) -> float:
        """计算样本特征与训练集均值的平均绝对差异。

        Args:
            sample_values: 样本的各特征值列表

        Returns:
            float: 标准化后的差异均值（值越大表示偏离训练集越多）
        """
        diffs = [
            abs(sample_values[i] - self.train_stats['features'][f]['mean'])
            for i, f in enumerate(self.features)
        ]
        return np.mean(diffs)

    
    def score_sample(
        self, 
        sample_idx: int, 
        unlabeled_stats: Dict, 
        top_n_indices: Set[int]
    ) -> Tuple[str, float]:
        """计算单个样本的综合选择评分。

        Args:
            sample_idx: 样本在未标注池中的索引
            unlabeled_stats: 未标注池统计信息
            top_n_indices: 高 SSC 分数样本的索引集合

        Returns:
            Tuple[str, float]: (样本名称, 综合评分)

        Scoring Formula:
            total_score = ssc_weight * (
                0.5 * mean(wasserstein_distances) +
                0.3 * balance_score +
                0.2 * difference_score
            )
            其中 ssc_weight = 1.5（如果样本在 top_n_indices 中）否则 1.0
        """
        image_name = unlabeled_stats['image_mapping']['image_names'][sample_idx]
        values = [
            unlabeled_stats['features'][f]['distribution'][sample_idx]
            for f in self.features
        ]
        
        wasserstein_scores = [self._calculate_wasserstein(f, v) for f, v in zip(self.features, values)]
        balance_score = self._calculate_balance_score(values)
        diff_score = self._calculate_difference_score(values)
        
        ssc_weight = 1.5 if sample_idx in top_n_indices else 1.0
        total_score = ssc_weight * (
            0.5 * np.mean(wasserstein_scores) +
            0.3 * balance_score +
            0.2 * diff_score
        )
        return (image_name, total_score)
    
    def select_samples(
        self, 
        unlabeled_stats: Dict, 
        select_num: int = 5, 
        ssc_top_percent: float = 0.8
    ) -> List[str]:
        """执行样本选择流程。

        Args:
            unlabeled_stats: 未标注池统计信息，需包含:
                - features: 各特征的分布
                - image_mapping: 样本名称与索引的映射
            select_num: 需选择的样本数量
            ssc_top_percent: 预筛选 SSC top 样本的百分比（默认 80%）

        Returns:
            List[str]: 选中的样本名称列表

        Steps:
            1. 预筛选 SSC 分数前 top_percent% 的样本
            2. 计算候选样本的综合评分
            3. 按评分排序并返回最优样本
        """
        ssc_scores = unlabeled_stats['features']['ssc_score']['distribution']
        threshold = np.percentile(ssc_scores, 100 * (1 - ssc_top_percent))
        candidate_indices = [i for i, score in enumerate(ssc_scores) if score >= threshold]
        
        top_n = int(len(candidate_indices) * ssc_top_percent)
        top_n_indices = set(sorted(
            candidate_indices, 
            key=lambda i: ssc_scores[i], 
            reverse=True
        )[:top_n])
        
        scored_samples = [
            self.score_sample(idx, unlabeled_stats, top_n_indices)
            for idx in candidate_indices
        ]
        scored_samples.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scored_samples[:select_num]]
    
    def get_selection_report(
        self, 
        selected_ids: List[str], 
        unlabeled_stats: Dict
    ) -> Dict:
        """生成选中样本的统计分析报告。

        Args:
            selected_ids: 选中的样本名称列表
            unlabeled_stats: 未标注池统计信息

        Returns:
            Dict: 包含以下键的报告字典:
                - selected_features: 各特征在选中样本中的值列表
                - wasserstein_changes: 各特征的 Wasserstein 距离变化

        Example:
            {
                'selected_features': {
                    'occlusion_score': [0.1, 0.3, ...],
                    ...
                },
                'wasserstein_changes': {
                    'occlusion_score': 0.05,
                    ...
                }
            }
        """
        report = {'selected_features': {}, 'wasserstein_changes': {}}
        selected_indices = [
            unlabeled_stats['image_mapping']['indices'][img_id]
            for img_id in selected_ids
        ]
        
        for feature in self.features:
            report['selected_features'][feature] = [
                unlabeled_stats['features'][feature]['distribution'][idx]
                for idx in selected_indices
            ]
            original_dist = np.array(self.train_stats['features'][feature]['distribution'])
            new_dist = np.concatenate([original_dist, report['selected_features'][feature]])
            report['wasserstein_changes'][feature] = wasserstein_distance(
                original_dist, new_dist)
        
        return report