from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CombinatorialSelector:
    def __init__(self, feature_weights=None, ssc_weight=0.7):
        """
        初始化组合优化选择器
        Args:
            feature_weights: 特征权重字典
            ssc_weight: SSC得分权重
        """
        # 确保feature_weights是字典类型
        if feature_weights is not None and not isinstance(feature_weights, dict):
            logger.warning("Invalid feature_weights format, using default weights")
            feature_weights = None
            
        self.feature_weights = feature_weights or {
            'occlusion_score': 0.3,
            'crown_count_score': 0.25,
            'diversity_score': 0.2,
            'area_var_score': 0.15,
            'density_var_score': 0.1
        }
        self.ssc_weight = float(ssc_weight)  # 确保是浮点数
        
        logger.info(f"Initialized CombinatorialSelector with weights: {self.feature_weights}")
        logger.info(f"SSC weight: {self.ssc_weight}")

    def select_samples(self, train_stats, unlabeled_stats, select_num=5):
        """样本选择主方法"""
        try:
            # 验证输入数据
            if not self._validate_input(train_stats, unlabeled_stats):
                raise ValueError("Invalid input data format")

            # 调用选择方法
            selected_indices, report = self._combinatorial_selection(
                train_stats,
                unlabeled_stats,
                select_num
            )
            
            # 保存报告用于后续查询
            self._last_report = report
            
            # 返回选中的样本名称
            selected_names = [unlabeled_stats['image_mapping']['image_names'][i] 
                            for i in selected_indices]
            return selected_names, report
            
        except Exception as e:
            logger.error(f"Error in sample selection: {str(e)}")
            raise

    def get_selection_report(self, selected_samples, unlabeled_stats):
        """生成选择报告"""
        return self._last_report if hasattr(self, '_last_report') else {}

    def _validate_input(self, train_stats, unlabeled_stats):
        """验证输入数据格式"""
        required_features = set([
            'ssc_score', 'occlusion_score', 'crown_count_score',
            'diversity_score', 'area_var_score', 'density_var_score'
        ])
        
        try:
            # 检查训练集统计信息
            if not all(f in train_stats['features'] for f in required_features):
                logger.error("Missing required features in train_stats")
                return False
                
            # 检查未标注池统计信息
            if not all(f in unlabeled_stats['features'] for f in required_features):
                logger.error("Missing required features in unlabeled_stats")
                return False
                
            # 检查图片映射信息
            if 'image_mapping' not in unlabeled_stats:
                logger.error("Missing image_mapping in unlabeled_stats")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            return False

    def _combinatorial_selection(self, train_stats, unlabeled_stats, select_num):
        """组合优化选择实现"""
        # 数据准备
        ssc_scores = np.array(unlabeled_stats['features']['ssc_score']['distribution'])
        feature_names = list(self.feature_weights.keys())
        
        # 创建优化问题
        prob = LpProblem("EnhancedActiveLearning", LpMaximize)
        
        # 定义决策变量
        n_samples = len(ssc_scores)
        x = [LpVariable(f"x_{i}", cat='Binary') for i in range(n_samples)]
        
        try:
            # ========== 目标函数 ==========
            # 1. SSC得分部分
            ssc_part = lpSum([x[i] * ssc_scores[i] for i in range(n_samples)])
            
            # 2. 特征分布匹配部分
            distribution_penalty = 0
            for f_name in feature_names:
                # 训练集特征统计量
                train_mean = train_stats['features'][f_name]['mean']
                train_std = train_stats['features'][f_name]['std'] + 1e-6
                
                # 未标注池特征值
                unlabeled_values = np.array(unlabeled_stats['features'][f_name]['distribution'])
                
                # 标准化差异
                normalized_diff = [(unlabeled_values[i] - train_mean)/train_std 
                                for i in range(n_samples)]
                
                # 加权惩罚
                weight = self.feature_weights[f_name]
                distribution_penalty += weight * lpSum([x[i] * abs(normalized_diff[i]) 
                                                    for i in range(n_samples)])
            
            # 综合目标函数
            prob += self.ssc_weight * ssc_part - (1 - self.ssc_weight) * distribution_penalty
            
            # ========== 约束条件 ==========
            prob += lpSum(x) == select_num
            
            # 求解问题
            status = prob.solve()
            if status != 1:
                raise RuntimeError("Failed to solve optimization problem")
            
            # 结果解析
            selected_indices = [i for i in range(n_samples) if x[i].value() == 1]
            
            # 生成报告
            report = {
                'selected_count': len(selected_indices),
                'avg_ssc': float(np.mean(ssc_scores[selected_indices])),
                'feature_changes': {},
                'optimization_status': status
            }
            
            # 计算特征变化
            for f_name in feature_names:
                original_mean = train_stats['features'][f_name]['mean']
                new_mean = np.mean([unlabeled_stats['features'][f_name]['distribution'][i] 
                                for i in selected_indices])
                report['feature_changes'][f_name] = {
                    'change': float(new_mean - original_mean),
                    'change_pct': float((new_mean - original_mean) / (original_mean + 1e-6) * 100)
                }
            
            return selected_indices, report
            
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            raise