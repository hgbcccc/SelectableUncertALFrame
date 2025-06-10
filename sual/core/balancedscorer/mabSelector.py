# sual/core/balancedscorer/mabSelector.py

import numpy as np
from typing import Dict, List, Optional
import torch

class MABSelector:
    """Multi-Armed Bandit based selector for active learning
    
    将样本选择问题建模为多臂老虎机问题:
    - 每个样本视为一个老虎机实例
    - 主臂为ssc_score (主要奖励)
    - 副臂为5个特征指标 (平衡奖励)
    """
    
    def __init__(self,
                 select_num: int,
                 ssc_weight: float = 0.7,
                 feature_weights: Optional[List[float]] = None,
                 exploration_rate: float = 0.1,
                 temperature: float = 1.0):
        """
        Args:
            select_num: 需要选择的样本数量
            ssc_weight: SSC分数的权重
            feature_weights: 5个特征指标的权重列表
            exploration_rate: epsilon-greedy策略的探索率
            temperature: 用于调整tanh函数的敏感度
        """
        self.select_num = select_num
        self.ssc_weight = ssc_weight
        
        # 特征权重默认均匀分配
        if feature_weights is None:
            self.feature_weights = [(1 - ssc_weight) / 5] * 5
        else:
            assert len(feature_weights) == 5
            assert abs(sum(feature_weights) - (1 - ssc_weight)) < 1e-6
            self.feature_weights = feature_weights
            
        self.exploration_rate = exploration_rate
        self.temperature = temperature
        
        # 用于记录选择过程的信息
        self.selection_info = {}
        
    def normalize_score(self, score: np.ndarray, stats: Dict) -> np.ndarray:
        """归一化分数到[0,1]区间"""
        min_val = stats['min']
        max_val = stats['max']
        if max_val == min_val:
            return np.ones_like(score)
        return (score - min_val) / (max_val - min_val)
    
    def calculate_feature_balance(self,
                                unlabeled_features: Dict,
                                train_stats: Dict) -> np.ndarray:
        """计算特征平衡奖励
        
        Args:
            unlabeled_features: 未标注样本的特征
            train_stats: 训练集的统计信息
            
        Returns:
            特征平衡奖励分数 [N]
        """
        feature_names = [
            'occlusion_score',
            'crown_count_score',
            'diversity_score',
            'area_var_score',
            'density_var_score'
        ]
        
        balance_scores = []
        for idx, feat_name in enumerate(feature_names):
            # 获取当前特征的统计信息
            train_mean = train_stats['features'][feat_name]['mean']
            train_std = train_stats['features'][feat_name]['std']
            
            # 计算未标注样本与训练集均值的偏差
            deviation = np.abs(unlabeled_features[feat_name] - train_mean)
            normalized_dev = deviation / (train_std + 1e-6)
            
            # 使用tanh将偏差转换为相似度分数
            similarity = 1 - np.tanh(normalized_dev / self.temperature)
            balance_scores.append(similarity * self.feature_weights[idx])
            
        return np.sum(balance_scores, axis=0)
    
    def select_samples(self,
                      unlabeled_stats: Dict,
                      train_stats: Dict) -> np.ndarray:
        """选择样本
        
        Args:
            unlabeled_stats: 未标注池的统计信息
            train_stats: 训练集的统计信息
            
        Returns:
            选中的样本索引数组
        """
        # 获取SSC分数并归一化
        ssc_scores = unlabeled_stats['features']['ssc_score']
        normalized_ssc = self.normalize_score(
            ssc_scores,
            train_stats['features']['ssc_score']
        )
        
        # 计算特征平衡奖励
        balance_rewards = self.calculate_feature_balance(
            unlabeled_stats['features'],
            train_stats
        )
        
        # 计算总奖励
        total_rewards = (self.ssc_weight * normalized_ssc +
                        (1 - self.ssc_weight) * balance_rewards)
        
        # 记录选择信息
        self.selection_info = {
            'ssc_scores': normalized_ssc,
            'balance_rewards': balance_rewards,
            'total_rewards': total_rewards
        }
        
        # epsilon-greedy策略
        if np.random.random() < self.exploration_rate:
            # 探索：随机选择
            selected_indices = np.random.choice(
                len(total_rewards),
                size=self.select_num,
                replace=False
            )
        else:
            # 利用：选择奖励最高的样本
            selected_indices = np.argsort(total_rewards)[-self.select_num:]
        
        return selected_indices
    
    def get_selection_report(self) -> Dict:
        """获取选择报告"""
        report = {
            'selector_type': 'MABSelector',
            'parameters': {
                'ssc_weight': self.ssc_weight,
                'feature_weights': self.feature_weights,
                'exploration_rate': self.exploration_rate,
                'temperature': self.temperature,
                'select_num': self.select_num
            },
            'selection_stats': {
                'mean_ssc': float(np.mean(self.selection_info['ssc_scores'])),
                'mean_balance': float(np.mean(self.selection_info['balance_rewards'])),
                'mean_total': float(np.mean(self.selection_info['total_rewards'])),
                'selected_rewards': float(np.mean(
                    self.selection_info['total_rewards'][self.last_selected_indices]
                )) if hasattr(self, 'last_selected_indices') else None
            }
        }
        return report