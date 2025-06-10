import numpy as np
from typing import Dict, List, Tuple
import random
from scipy.stats import wasserstein_distance
from scipy.special import kl_div, rel_entr
import os
import json

class RLSampleSelector:
    """基于强化学习的主动学习样本选择器，支持多种分布差异度量方法"""
    
    def __init__(self, train_stats: Dict, metric: str = 'wasserstein'):
        """
        初始化选择器
        
        Args:
            train_stats: 训练集统计信息
            metric: 分布差异度量方法，可选值:
                - 'wasserstein': Wasserstein距离
                - 'kl': KL散度
                - 'js': JS散度
                - 'euclidean': 欧几里得距离
                - 'simple_stats': 简单统计量差异
                - 'percentile': 百分位数差异
                - 'ensemble': 集成多种方法
        """
        self.train_stats = train_stats
        self.features = [
            'occlusion_score',
            'crown_count_score',
            'diversity_score',
            'area_var_score',
            'density_var_score'
        ]
        # 设置度量方法
        self.metric = metric
        
        # RL参数
        self.alpha = 0.1      # 学习率
        self.gamma = 0.9      # 折扣因子
        self.epsilon = 0.1    # 探索率
        
        # 超参数权重
        self.weights = {
            'ssc': 0.4,           # SSC分数权重
            'balance': 0.4,       # 特征平衡权重
            'distribution': 0.2   # 分布变化权重
        }
        
        # 初始化Q表
        self.q_table = {}
        
        # 创建方法字典
        self.metric_methods = {
            'wasserstein': self._wasserstein_distance,
            'kl': self._kl_divergence,
            'js': self._js_divergence,
            'euclidean': self._euclidean_distance,
            'simple_stats': self._simple_stats_diff,
            'percentile': self._percentile_diff,
            'ensemble': self._ensemble_metrics
        }
        
        # 验证度量方法是否有效
        if metric not in self.metric_methods:
            raise ValueError(f"不支持的度量方法: {metric}，有效选项: {list(self.metric_methods.keys())}")
    
    def _wasserstein_distance(self, feature: str, sample_value: float) -> float:
        """计算Wasserstein距离"""
        original_dist = np.array(self.train_stats['features'][feature]['distribution'])
        new_dist = np.append(original_dist, sample_value)
        return wasserstein_distance(original_dist, new_dist)
    
    def _kl_divergence(self, feature: str, sample_value: float) -> float:
        """计算KL散度"""
        original_dist = np.array(self.train_stats['features'][feature]['distribution'])
        
        # 创建包含样本值的新分布
        new_dist = np.append(original_dist, sample_value)
        
        # 创建直方图分布
        bins = 10
        hist_orig, bin_edges = np.histogram(original_dist, bins=bins, density=True)
        hist_new, _ = np.histogram(new_dist, bins=bin_edges, density=True)
        
        # 避免0值
        hist_orig = hist_orig + 1e-10
        hist_new = hist_new + 1e-10
        
        # 归一化
        hist_orig = hist_orig / np.sum(hist_orig)
        hist_new = hist_new / np.sum(hist_new)
        
        # 计算KL散度
        kl = np.sum(rel_entr(hist_new, hist_orig))
        return kl
    
    def _js_divergence(self, feature: str, sample_value: float) -> float:
        """计算JS散度 (Jensen-Shannon divergence)"""
        original_dist = np.array(self.train_stats['features'][feature]['distribution'])
        
        # 创建包含样本值的新分布
        new_dist = np.append(original_dist, sample_value)
        
        # 创建直方图分布
        bins = 10
        hist_orig, bin_edges = np.histogram(original_dist, bins=bins, density=True)
        hist_new, _ = np.histogram(new_dist, bins=bin_edges, density=True)
        
        # 避免0值
        hist_orig = hist_orig + 1e-10
        hist_new = hist_new + 1e-10
        
        # 归一化
        hist_orig = hist_orig / np.sum(hist_orig)
        hist_new = hist_new / np.sum(hist_new)
        
        # 计算混合分布
        mixed = 0.5 * (hist_orig + hist_new)
        
        # 计算JS散度: 0.5 * (KL(P||M) + KL(Q||M))
        js = 0.5 * (np.sum(rel_entr(hist_orig, mixed)) + np.sum(rel_entr(hist_new, mixed)))
        return js
    
    def _euclidean_distance(self, feature: str, sample_value: float) -> float:
        """计算欧几里得距离"""
        # 使用样本值与训练集均值的欧几里得距离
        mean = self.train_stats['features'][feature]['mean']
        return np.abs(sample_value - mean) / (self.train_stats['features'][feature]['std'] + 1e-10)
    
    def _simple_stats_diff(self, feature: str, sample_value: float) -> float:
        """计算简单统计量差异"""
        train_mean = self.train_stats['features'][feature]['mean']
        train_std = self.train_stats['features'][feature]['std']
        
        # 标准化差异
        z_score = abs(sample_value - train_mean) / (train_std + 1e-10)
        return z_score
    
    def _percentile_diff(self, feature: str, sample_value: float) -> float:
        """计算百分位数差异"""
        original_dist = np.array(self.train_stats['features'][feature]['distribution'])
        percentiles = [25, 50, 75]
        
        # 计算原始分布的百分位数
        orig_percentiles = np.percentile(original_dist, percentiles)
        
        # 创建包含样本值的新分布
        new_dist = np.append(original_dist, sample_value)
        new_percentiles = np.percentile(new_dist, percentiles)
        
        # 计算百分位数变化的平均绝对值
        percentile_diff = np.mean(np.abs(new_percentiles - orig_percentiles))
        return percentile_diff
    
    def _ensemble_metrics(self, feature: str, sample_value: float) -> float:
        """集成多种度量方法"""
        # 排除当前方法和ensemble避免递归
        methods = [method for name, method in self.metric_methods.items() 
                   if name not in ['ensemble']]
        
        # 计算所有方法的平均值
        scores = [method(feature, sample_value) for method in methods]
        # 归一化每个分数
        normalized_scores = [(s - min(scores)) / (max(scores) - min(scores) + 1e-10) 
                            for s in scores]
        return np.mean(normalized_scores)
    
    def _calculate_feature_balance_score(self, feature_values: List[float]) -> float:
        """计算特征平衡分数"""
        # 归一化特征值
        normalized_values = []
        for i, value in enumerate(feature_values):
            feature_name = self.features[i]
            feature_min = self.train_stats['features'][feature_name]['min']
            feature_max = self.train_stats['features'][feature_name]['max']
            # 避免除零
            if feature_max - feature_min < 1e-6:
                normalized = 0.0
            else:
                normalized = (value - feature_min) / (feature_max - feature_min + 1e-6)
            normalized_values.append(normalized)
        
        # 计算标准差，标准差越小表示越平衡
        std = np.std(normalized_values)
        # 平衡分数 = 1 / (1 + 标准差)，使得标准差越小，分数越高
        balance_score = 1.0 / (1.0 + std)
        return balance_score
    
    def _calculate_distribution_change(self, unlabeled_stats: Dict, sample_idx: int) -> float:
        """计算分布变化分数"""
        # 使用选择的度量方法计算每个特征的分布变化
        scores = []
        for feature in self.features:
            sample_value = unlabeled_stats['features'][feature]['distribution'][sample_idx]
            method = self.metric_methods[self.metric]
            score = method(feature, sample_value)
            scores.append(score)
        
        # 返回平均分布变化分数
        return np.mean(scores)
    
    def _calculate_reward(self, unlabeled_stats: Dict, sample_idx: int) -> float:
        """计算选择一个样本的奖励"""
        # 1. 获取样本特征值
        feature_values = [
            unlabeled_stats['features'][f]['distribution'][sample_idx] 
            for f in self.features
        ]
        
        # 2. 获取样本SSC分数
        ssc_score = unlabeled_stats['features']['ssc_score']['distribution'][sample_idx]
        
        # 3. 特征平衡分数
        balance_score = self._calculate_feature_balance_score(feature_values)
        
        # 4. 分布变化分数
        distribution_change = self._calculate_distribution_change(unlabeled_stats, sample_idx)
        
        # 5. 归一化SSC分数
        max_ssc = unlabeled_stats['features']['ssc_score']['max']
        min_ssc = unlabeled_stats['features']['ssc_score']['min']
        normalized_ssc = (ssc_score - min_ssc) / (max_ssc - min_ssc + 1e-6)
        
        # 6. 计算总奖励
        total_reward = (
            self.weights['ssc'] * normalized_ssc +
            self.weights['balance'] * balance_score +
            self.weights['distribution'] * distribution_change
        )
        
        return total_reward
    
    def _build_q_table(self, unlabeled_stats: Dict, candidate_indices: List[int]) -> Dict:
        """构建Q表"""
        q_table = {}
        for idx in candidate_indices:
            reward = self._calculate_reward(unlabeled_stats, idx)
            q_table[idx] = reward
        return q_table
    
    def _select_action(self, q_table: Dict, selected_indices: List[int]) -> int:
        """使用epsilon-greedy策略选择动作"""
        # 获取未选择的候选索引
        available_indices = [idx for idx in q_table.keys() if idx not in selected_indices]
        
        if not available_indices:
            return None
        
        # epsilon-greedy策略
        if random.random() < self.epsilon:
            # 随机探索
            return random.choice(available_indices)
        else:
            # 贪心选择
            max_q = max([q_table[idx] for idx in available_indices])
            best_actions = [idx for idx in available_indices if q_table[idx] == max_q]
            return random.choice(best_actions)
    
    def save_q_table(self, path: str) -> None:
        """保存Q表到文件"""
        # 将索引转换为字符串，因为JSON不支持整数作为键
        serializable_q_table = {str(k): v for k, v in self.q_table.items()}
        
        with open(path, 'w') as f:
            json.dump(serializable_q_table, f, indent=2)
    
    def load_q_table(self, path: str) -> None:
        """从文件加载Q表"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                serialized_q_table = json.load(f)
                # 将键转换回整数
                self.q_table = {int(k): v for k, v in serialized_q_table.items()}
    
    def select_samples(
        self, 
        unlabeled_stats: Dict, 
        select_num: int = 5,
        ssc_top_percent: float = 0.8,
        q_table_path: str = None
    ) -> List[str]:
        """
        使用RL选择样本
        
        Args:
            unlabeled_stats: 未标注样本统计信息
            select_num: 需要选择的样本数量
            ssc_top_percent: 预筛选SSC top百分比
            q_table_path: Q表保存/加载路径，如果提供则尝试加载
            
        Returns:
            选中的样本ID列表
        """
        # 1. 预筛选候选样本
        ssc_scores = unlabeled_stats['features']['ssc_score']['distribution']
        threshold = np.percentile(ssc_scores, 100 * (1 - ssc_top_percent))
        candidate_indices = [
            i for i, score in enumerate(ssc_scores)
            if score >= threshold
        ]
        
        # 如果候选样本少于需要选择的数量，直接返回所有候选
        if len(candidate_indices) <= select_num:
            return [unlabeled_stats['image_mapping']['image_names'][idx] for idx in candidate_indices]
        
        # 2. 尝试加载Q表
        if q_table_path and os.path.exists(q_table_path):
            self.load_q_table(q_table_path)
        
        # 3. 构建或更新Q表
        current_q_table = self._build_q_table(unlabeled_stats, candidate_indices)
        
        # 合并到全局Q表
        for idx, value in current_q_table.items():
            if idx not in self.q_table:
                self.q_table[idx] = value
            else:
                # 更新现有值
                self.q_table[idx] = (1 - self.alpha) * self.q_table[idx] + self.alpha * value
        
        # 4. 选择样本
        selected_indices = []
        for _ in range(select_num):
            action = self._select_action(current_q_table, selected_indices)
            if action is None:
                break
            selected_indices.append(action)
        
        # 5. 保存Q表
        if q_table_path:
            self.save_q_table(q_table_path)
        
        # 6. 返回选择的样本ID
        return [unlabeled_stats['image_mapping']['image_names'][idx] for idx in selected_indices]
    
    def get_selection_report(self, selected_ids: List[str], unlabeled_stats: Dict) -> Dict:
        """生成选择报告"""
        report = {
            'selected_features': {},
            'distribution_changes': {},
            'balance_scores': {},
            'method': f'RL-based selection with {self.metric} metric'
        }
        
        # 获取选中样本的索引
        selected_indices = [
            unlabeled_stats['image_mapping']['indices'][img_id]
            for img_id in selected_ids
        ]
        
        # 所有特征，包括ssc_score
        all_features = self.features + ['ssc_score']
        
        for feature in all_features:
            # 选中的特征值
            report['selected_features'][feature] = [
                unlabeled_stats['features'][feature]['distribution'][idx]
                for idx in selected_indices
            ]
            
            # 如果是核心特征，计算分布变化
            if feature in self.features:
                # 使用当前选择的度量方法
                method = self.metric_methods[self.metric]
                
                # 计算每个样本的分布变化
                changes = []
                for idx in selected_indices:
                    sample_value = unlabeled_stats['features'][feature]['distribution'][idx]
                    change = method(feature, sample_value)
                    changes.append(change)
                
                report['distribution_changes'][feature] = {
                    'mean': np.mean(changes),
                    'values': changes
                }
        
        # 计算每个样本的特征平衡分数
        for i, idx in enumerate(selected_indices):
            feature_values = [
                unlabeled_stats['features'][f]['distribution'][idx]
                for f in self.features
            ]
            report['balance_scores'][selected_ids[i]] = self._calculate_feature_balance_score(feature_values)
        
        # 添加平均特征平衡分数
        report['avg_balance_score'] = np.mean(list(report['balance_scores'].values()))
        
        # 添加度量方法信息
        report['metric'] = self.metric
        
        return report