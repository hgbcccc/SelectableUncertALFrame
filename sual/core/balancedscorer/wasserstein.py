
import numpy as np
from typing import Dict, List
from scipy.stats import wasserstein_distance

class WassersteinBalancedScorer:
    """基于Wasserstein距离的平衡评分器"""
    """Wasserstein距离衡量的是将一个分布“搬运”到另一个分布所需的最小“工作量”。"""
    
    def __init__(self,
                alpha: float = 0.5,
                beta: float = 0.3,
                gamma: float = 0.2,
                mapd_threshold: float = 0.25,
                train_stats: Dict = None,
                feature_names: List[str] = [
                    'occlusion_score', 'crown_count_score', 'diversity_score',
                    'area_var_score', 'density_var_score'
                ]):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mapd_threshold = mapd_threshold
        self.feature_names = feature_names
        self.train_stats = train_stats if train_stats is not None else {
            'features': {},
            'batch_statistics': {}
        }

    def _standardize_features(self, features: Dict[str, float]) -> np.ndarray:
        """标准化特征
        
        Args:
            features: 特征字典
                
        Returns:
            标准化后的特征数组
        """
        try:
            # 提取特征值
            feature_values = np.array([features[name] for name in self.feature_names])
            
            # 从train_stats中提取每个特征的均值和标准差
            means = np.array([self.train_stats['features'][name]['mean'] 
                            for name in self.feature_names])
            stds = np.array([self.train_stats['features'][name]['std'] 
                            for name in self.feature_names])
            
            # 标准化
            standardized = (feature_values - means) / (stds + 1e-6)
            
            print(f"Feature values: {feature_values}")
            print(f"Means: {means}")
            print(f"Stds: {stds}")
            print(f"Standardized: {standardized}")
            
            return standardized
            
        except Exception as e:
            print(f"标准化特征时出错: {e}")
            raise  # 重新抛出异常，以便调用者知道发生了错误

    def _calculate_mapd(self, standardized_features: np.ndarray) -> float:
        """计算平均绝对特征差距(MAPD)"""
        n = len(standardized_features)
        diffs = []
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(abs(standardized_features[i] - standardized_features[j]))
        return np.mean(diffs) if diffs else 0.0
    
    def _calculate_wasserstein_distances(self, features: Dict[str, float]) -> Dict[str, float]:
        """计算每个特征的Wasserstein距离"""
        distances = {}
        standardized = self._standardize_features(features)
        
        for i, name in enumerate(self.feature_names):
            # 获取训练集中该特征的分布
            if ('features' in self.train_stats and 
                name in self.train_stats['features'] and 
                'distribution' in self.train_stats['features'][name]):
                
                train_distribution = self.train_stats['features'][name]['distribution']
                
                # 如果训练集分布为空，返回0
                if not train_distribution:
                    distances[name] = 0.0
                    continue
                
                # 标准化训练集分布
                train_mean = self.train_stats['features'][name]['mean']
                train_std = self.train_stats['features'][name]['std']
                standardized_train = [(x - train_mean) / (train_std + 1e-6) for x in train_distribution]
                
                # 计算Wasserstein距离
                distances[name] = wasserstein_distance(
                    standardized_train,
                    [standardized[i]]  # 当前样本的标准化值
                )
            else:
                distances[name] = 0.0
                
        return distances
    
    def compute_balanced_score(self, uncertainty_metrics: Dict[str, float]) -> Dict:
        """计算平衡后的得分"""
        try:
            print("调用 compute_balanced_score 方法")
            # 1. 提取特征值
            features = {name: uncertainty_metrics[name] for name in self.feature_names}
            print(f"提取的特征: {features}")
            
            # 2. 标准化特征
            standardized_features = self._standardize_features(features)
            print(f"标准化后的特征: {standardized_features}")
            
            # 3. 计算线性组合得分(w)
            w_score = np.sum(standardized_features)
            print(f"w_score: {w_score}")
            
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
                'wasserstein_balanced_score': final_score * 100,
                'w_score': w_score,
                'mapd_score': mapd_score,
                'perturbation_score': perturbation_score,
                'feature_wasserstein_distances': w_distances
            })
            
            return result
            
        except Exception as e:
            print(f"计算平衡得分时出错: {e}")
            raise  # 重新抛出异常，以便调用者知道发生了错误

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
    





import numpy as np
from scipy.stats import wasserstein_distance
from typing import Dict, List, Tuple

class WassersteinSampleSelector:
    """基于Wasserstein距离的主动学习样本选择器"""
    
    def __init__(self, train_stats: Dict):
        """
        初始化选择器
        
        Args:
            train_stats: 训练集统计信息，包含各特征的分布数据
                        格式示例：
                        {
                            'features': {
                                'occlusion_score': {
                                    'mean': float,
                                    'std': float,
                                    'distribution': List[float]
                                },
                                # 其他特征...
                            },
                            'batch_statistics': Dict
                        }
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
        """计算单个特征的Wasserstein距离"""
        original_dist = np.array(self.train_stats['features'][feature]['distribution'])
        new_dist = np.append(original_dist, new_value)
        return wasserstein_distance(original_dist, new_dist)
    
    def _calculate_balance_score(self, feature_values: List[float]) -> float:
        """计算五个特征的平衡性得分"""
        cv = np.std(feature_values) / (np.mean(feature_values) + 1e-6)
        return 1 / (1 + cv)
    
    def _calculate_difference_score(self, sample_values: List[float]) -> float:
        """计算与训练集平均水平的差异度"""
        diffs = [
            abs(sample_values[i] - self.train_stats['features'][f]['mean'])
            for i, f in enumerate(self.features)
        ]
        return np.mean(diffs)
    
    def score_sample(self, sample_id: str, sample_data: Dict, top_n_ids: set) -> Tuple[str, float]:
        """
        计算单个样本的综合评分
        
        Args:
            sample_id: 样本ID
            sample_data: 样本数据（需包含uncertainty字典）
            top_n_ids: SSC top n样本ID集合
            
        Returns:
            (sample_id, total_score)
        """
        # 1. 提取特征值
        values = [sample_data['uncertainty'][f] for f in self.features]
        
        # 2. 计算三项指标
        wasserstein_scores = [self._calculate_wasserstein(f, v) for f, v in zip(self.features, values)]
        balance_score = self._calculate_balance_score(values)
        diff_score = self._calculate_difference_score(values)
        
        # 3. SSC权重
        ssc_weight = 1.5 if sample_id in top_n_ids else 1.0
        
        # 4. 综合评分
        total_score = ssc_weight * (
            0.5 * np.mean(wasserstein_scores) +  # 分布变化
            0.3 * balance_score +               # 特征平衡
            0.2 * diff_score                    # 差异程度
        )
        
        return (sample_id, total_score)
    
    def select_samples(
        self, 
        unlabeled_pool: Dict, 
        select_num: int = 5,
        ssc_top_percent: float = 0.8
    ) -> List[str]:
        """
        执行样本选择
        
        Args:
            unlabeled_pool: 未标注样本池，格式：
                            {
                                'img_id.jpg': {
                                    'uncertainty': {
                                        'ssc_score': float,
                                        'occlusion_score': float,
                                        # 其他特征...
                                    }
                                },
                                # 更多样本...
                            }
            select_num: 需要选择的样本数量
            ssc_top_percent: 预筛选SSC top百分比
            
        Returns:
            选中的样本ID列表，如：
            ['img1.jpg', 'img2.jpg', ...]
        """
        # 1. 预筛选SSC top样本
        all_ssc = [v['uncertainty']['ssc_score'] for v in unlabeled_pool.values()]
        threshold = np.percentile(all_ssc, 100 * (1 - ssc_top_percent))
        candidates = {
            k: v for k, v in unlabeled_pool.items()
            if v['uncertainty']['ssc_score'] >= threshold
        }
        
        # 2. 确定SSC top样本ID（用于加权）
        top_n = int(len(candidates) * ssc_top_percent)
        top_n_ids = set(sorted(
            candidates.keys(),
            key=lambda x: candidates[x]['uncertainty']['ssc_score'],
            reverse=True
        )[:top_n])
        
        # 3. 计算所有候选样本评分
        scored_samples = [
            self.score_sample(sample_id, sample_data, top_n_ids)
            for sample_id, sample_data in candidates.items()
        ]
        
        # 4. 排序并选择
        scored_samples.sort(key=lambda x: x[1], reverse=True)
        actual_select_num = min(select_num, len(scored_samples))  # 添加这行
        return [x[0] for x in scored_samples[:actual_select_num]]  # 修改这行
    
    def get_selection_report(self, selected_ids: List[str], unlabeled_pool: Dict) -> Dict:
        """
        生成选择报告
        
        Returns:
            {
                'selected_features': {
                    'occlusion_score': [选中的样本值列表],
                    # 其他特征...
                },
                'wasserstein_changes': {
                    'occlusion_score': 平均Wasserstein变化,
                    # 其他特征...
                }
            }
        """
        report = {
            'selected_features': {},
            'wasserstein_changes': {}
        }
        
        for feature in self.features:
            # 选中的特征值
            report['selected_features'][feature] = [
                unlabeled_pool[sample_id]['uncertainty'][feature]
                for sample_id in selected_ids
            ]
            
            # Wasserstein变化
            original_dist = np.array(self.train_stats['features'][feature]['distribution'])
            new_values = report['selected_features'][feature]
            new_dist = np.concatenate([original_dist, new_values])
            report['wasserstein_changes'][feature] = wasserstein_distance(
                original_dist, new_dist)
        
        return report