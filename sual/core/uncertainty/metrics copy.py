# sual/core/uncertainty/metrics.py

import numpy as np
from typing import Dict, Union, List, Optional
from mmdet.structures import DetDataSample
import json
from pathlib import Path
from sual.core.uncertainty.measures import (
    basic_uncertainty,
    entropy_uncertainty,
    variance_uncertainty,
    quantile_uncertainty,
    density_uncertainty,
    compute_value_metrics,
    box_uncertainty,
    calculate_sor,  
    margin_uncertainty,
    least_confidence_uncertainty,
    calculate_ssc,
)

class UncertaintyMetrics:
    def __init__(self):
        self.metrics_mapping = {
            'basic': basic_uncertainty,
            'entropy': entropy_uncertainty,
            'variance': variance_uncertainty,
            'quantile': quantile_uncertainty,
            'density': density_uncertainty,
            'value': compute_value_metrics,
            'box': box_uncertainty,
            'sor': calculate_sor,  # 确保这里的映射正确
            'margin': margin_uncertainty,
            'least_confidence': least_confidence_uncertainty,
            'ssc': calculate_ssc  # 添加SSC方法的映射
        }
    
    def compute_uncertainty(self, 
                        result: DetDataSample,
                        methods: Union[str, List[str]] = 'all',
                        min_score_thresh: float = 0.1,
                        img_shape: Optional[tuple] = None) -> Dict[str, float]:
        """计算检测结果的不确定性"""
        # 添加调试信息
        # print(f"Computing uncertainty with methods: {methods}")
        # print(f"Has pred_instances: {hasattr(result, 'pred_instances')}")
        # if hasattr(result, 'pred_instances'):
        #     # print(f"Number of predictions: {len(result.pred_instances)}")
        #     if hasattr(result.pred_instances, 'scores'):
        #         print(f"Scores: {result.pred_instances.scores}")
        #         print(f"All scores shape: {result.pred_instances.all_scores.shape}")

        if not hasattr(result, 'pred_instances') or len(result.pred_instances) == 0:
            print("No predictions found")
            return self._get_empty_metrics()
            
        mask = result.pred_instances.scores >= min_score_thresh
        if not mask.any():
            return self._get_empty_metrics()
            
        # 处理 'all' 选项
        if methods == 'all':
            methods = list(self.metrics_mapping.keys())
        elif isinstance(methods, str):
            methods = [methods]

        metrics = {}
        for method in methods:
            if method in self.metrics_mapping:
                try:
                    if method == 'box':
                        method_metrics = self.metrics_mapping[method](result, img_shape)
                    else:
                        method_metrics = self.metrics_mapping[method](result)
                    # print(f"Method {method} returned metrics: {method_metrics}")
                    metrics.update(method_metrics)
                except Exception as e:
                    print(f"Error computing {method} uncertainty: {str(e)}")
                    continue

        return metrics

    def _get_empty_metrics(self) -> Dict[str, float]:
        """返回空的度量结果"""
        base_metrics = {
            'max_uncertainty': 0.0,
            'avg_uncertainty': 0.0,
            'sum_uncertainty': 0.0,
            'entropy': 0.0,
            'normalized_entropy': 0.0,
            'variance': 0.0,
            'std': 0.0,
            'cv': 0.0,
            'q25': 0.0,
            'q50': 0.0,
            'q75': 0.0,
            'iqr': 0.0,
            'high_uncertainty_ratio': 0.0,
            'uncertainty_density': 0.0,
            'difficulty_score': 0.0,
            'information_score': 0.0,
            'diversity_score': 0.0,
            'total_value_score': 0.0,
            'num_objects': 0,
            'num_classes': 0,
            'class_distribution': {},
            'margin': 0.0,
            'normalized_margin': 0.0,
            'least_confidence': 0.0,
            'mean_least_confidence': 0.0,


        }
        # 添加框相关的空度量
        box_metrics = {
            'iou_uncertainty': 0.0,
            'center_entropy': 0.0,
            'center_variance': 0.0
        }
        # 添加空间遮挡率(SOR)相关的空度量
        sor_metrics = {
            'max_sor': 0.0,
            'avg_sor': 0.0,
            'sum_sor': 0.0
        }
        # 添加SSC相关的空度量
        ssc_metrics = {
            'ssc': {  # 将所有SSC相关指标放在'ssc'子字典中
                'ssc_score': None,
                'occlusion_score': 0.0,
                'crown_count_score': None,
                'diversity_score': 0.0,
                'area_var_score': 0.0,
                'density_var_score': 0.0,
                'crown_count': 0
            }
        }
        return {**base_metrics, **box_metrics, **sor_metrics, **ssc_metrics}
    

    def rank_samples(self,
                    results: List[DetDataSample],
                    method: str = 'entropy',
                    selected_metric: str = 'normalized_entropy',  # 新增参数，选择具体指标来序列化
                    strategy: str = 'max',
                    min_score_thresh: float = 0.1,
                    weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """对样本进行不确定性排序
        
        Args:
            results (List[DetDataSample]): 检测结果列表
            method (str): 不确定性计算方法
            selected_metric (str): 选择的具体指标
            strategy (str): 排序策略 ('max' 或 'min')
            min_score_thresh (float): 最小置信度阈值
            weights (Dict[str, float], optional): 混合策略的权重
            
        Returns:
            np.ndarray: 排序后的样本索引
        """
        scores = []
        valid_indices = []
        
        # 定义每种方法下的可选指标
        method_metrics = {
            'basic': ['max_uncertainty', 'avg_uncertainty', 'sum_uncertainty'],
            'entropy': ['normalized_entropy', 'box_entropy', 'mean_entropy', 'class_entropy', 'normalized_class_entropy'],
            'variance': ['variance', 'std', 'cv'],
            'quantile': ['q25', 'q50', 'q75', 'iqr'],
            'density': ['uncertainty_density', 'high_uncertainty_ratio'],
            'value': ['total_value_score'],
            'box': ['iou_uncertainty', 'center_entropy', 'center_variance'],
            'sor': ['max_sor', 'avg_sor', 'sum_sor'],
            'margin': ['margin', 'mean_margin', 'min_margin'],
            'least_confidence': ['least_confidence', 'mean_least_confidence'],
            'ssc': ['ssc_score', 'occlusion_score', 'crown_count_score', 
                   'diversity_score', 'area_var_score', 'density_var_score']
        }

        for idx, result in enumerate(results):
            metrics = self.compute_uncertainty(result, method, min_score_thresh)
            
            # 根据选择的方法和指标获取对应的分数
            # 处理 'all' 方法
            if method == 'all':
                score = metrics.get(selected_metric, 0.0)  # 使用 selected_metric 进行排序
            else:
                if method in method_metrics and selected_metric in method_metrics[method]:
                    score = metrics.get(selected_metric, 0.0)
                else:
                    score = 0.0  # 如果方法或指标不在可用列表中，默认分数为0
            
            # 仅考虑有效样本
            if score > 0:  # 有效样本
                scores.append(score)
                valid_indices.append(idx)
        
        if not valid_indices:
            return np.array([])  # 如果没有有效样本，返回空数组
        
        # 排序
        scores = np.array(scores)
        valid_indices = np.array(valid_indices)
        if strategy == 'max':
            ranked_indices = valid_indices[np.argsort(scores)[::-1]]  # 从高到低排序
        else:
            ranked_indices = valid_indices[np.argsort(scores)]  # 从低到高排序
        
        return ranked_indices


    def analyze_image_uncertainty(self, 
                                img_path: str,
                                result: DetDataSample,
                                methods: Optional[List[str]] = None,
                                min_score_thresh: float = 0.3) -> Dict[str, Dict[str, float]]:
        """分析单张图片的不确定性
        
        Args:
            img_path (str): 图片路径
            result (DetDataSample): MMDetection的检测结果
            methods (List[str], optional): 要使用的度量方法列表
            min_score_thresh (float): 最小置信度阈值
            
        Returns:
            Dict[str, Dict[str, float]]: 以图片路径为键的不确定性度量结果
        """
            # 确保方法列表包含 SOR
        if methods is None:
            methods = ['all']  # 默认使用所有方法
        elif 'sor' not in methods:
            methods.append('sor')  # 确保包含 SOR
            
        metrics = self.compute_uncertainty(
            result, methods=methods, min_score_thresh=min_score_thresh)
        return {img_path: metrics}
    


class MUSCDBSampling(UncertaintyMetrics):
    """两阶段主动学习采样策略：MUS + CDB"""
    def __init__(self, class_names: Optional[List[str]] = None):
        super().__init__()
        self.metrics_mapping['mus'] = self.compute_mus
        self.class_names = class_names  # 可选：类别ID到名称的映射
    
    def compute_mus(self, result: DetDataSample, theta: float = 0.1) -> Dict[str, float]:
        """计算MUS得分，返回 {bbox_id: score}"""
        if not hasattr(result, 'pred_instances') or len(result.pred_instances) == 0:
            return {}
        pred = result.pred_instances
        scores = pred.scores.cpu().numpy()
        all_scores = pred.all_scores.cpu().numpy()
        img_uncertainty = 1.0 - np.mean(scores[scores > theta]) if np.any(scores > theta) else 1.0
        eps = 1e-10
        obj_uncertainties = -np.sum(all_scores * np.log(all_scores + eps), axis=1)
        return {f"bbox_{i}": float(img_uncertainty * obj_uncertainties[i]) for i in range(len(scores))}
    
    def get_labeled_counts(self, stats_path: Path) -> Dict[int, int]:
        """从stats.json中提取已标注类别分布（返回类别ID: 数量）"""
        with open(stats_path, 'r') as f:
            category_stats = json.load(f)["dataset_stats"]["category_stats"]
        return {int(stats["id"]): stats["labeled"] for stats in category_stats.values()}  # 假设stats包含id字段
    
    def calculate_cdb_budget(self, 
                            labeled_counts: Dict[int, int],
                            current_labels: List[str],
                            total_budget: int = 100) -> Dict[str, int]:
        """计算类别预算，返回 {class_name: quota}"""
        # 将current_labels转换为类别统计
        from collections import Counter
        current_class_counts = Counter(current_labels)
        unique_classes = list(current_class_counts.keys())
        
        if not unique_classes:
            return {}
        
        # 计算总的已标注数量
        total_labeled = sum(labeled_counts.values()) if labeled_counts else 1
        
        # 为每个类别计算beta值（越少标注的类别beta越大）
        class_betas = {}
        for cls_name in unique_classes:
            # 尝试将类别名称转换为ID进行查找
            try:
                cls_id = int(cls_name)
                labeled_count = labeled_counts.get(cls_id, 0)
            except (ValueError, TypeError):
                # 如果不能转换为int，使用字符串查找或默认为0
                labeled_count = 0
            
            # 计算beta值，避免除零
            if total_labeled > 0:
                beta = 1.0 - (labeled_count / total_labeled)
            else:
                beta = 1.0
            class_betas[cls_name] = max(beta, 0.1)  # 确保最小值
        
        # 计算权重
        exp_betas = np.array([np.exp(class_betas[cls]) for cls in unique_classes])
        total_exp = np.sum(exp_betas)
        
        if total_exp == 0:
            # 如果所有权重都是0，平均分配
            weight_per_class = 1.0 / len(unique_classes)
            class_weights = {cls: weight_per_class for cls in unique_classes}
        else:
            class_weights = {cls: exp_beta / total_exp 
                           for cls, exp_beta in zip(unique_classes, exp_betas)}
        
        # 分配预算
        cls_budget = {}
        allocated_budget = 0
        
        # 首先为每个类别分配最少1个样本
        for cls in unique_classes:
            cls_budget[cls] = 1
            allocated_budget += 1
        
        # 分配剩余预算
        remaining_budget = total_budget - allocated_budget
        if remaining_budget > 0:
            for cls in unique_classes:
                additional = int(round(class_weights[cls] * remaining_budget))
                cls_budget[cls] += additional
                allocated_budget += additional
        
        # 确保不超过总预算
        while allocated_budget > total_budget and any(v > 1 for v in cls_budget.values()):
            # 从预算最多的类别中减少
            max_cls = max(cls_budget.keys(), key=lambda x: cls_budget[x])
            if cls_budget[max_cls] > 1:
                cls_budget[max_cls] -= 1
                allocated_budget -= 1
        
        return cls_budget
    
    def rank_samples_with_cdb(self, 
                            results: List[DetDataSample],
                            labeled_counts: Dict[str, Dict[str, int]],  # 直接传入类别统计
                            img_names: List[str],  # 添加图片名称列表
                            total_budget: int = 3200,
                            theta: float = 0.1,
                            strategy: str = 'max') -> List[str]:
        """两阶段排序：MUS得分排序 + CDB预算筛选"""
        if not results:
            return []
        
        # 1. 计算MUS得分并收集类别标签
        all_samples = []
        for idx, result in enumerate(results):
            mus_scores = self.compute_mus(result, theta)
            if not mus_scores:
                continue
            pred = result.pred_instances
            img_name = img_names[idx] if idx < len(img_names) else f"img_{idx}"
            
            # 计算图片级别的MUS得分（所有bbox的平均值）
            img_mus_score = np.mean(list(mus_scores.values())) if mus_scores else 0.0
            
            # 获取图片中的主要类别（出现最多的类别）
            if len(pred.labels) > 0:
                unique_labels, counts = np.unique(pred.labels.cpu().numpy(), return_counts=True)
                main_class_id = unique_labels[np.argmax(counts)]
                class_name = str(main_class_id)  # 统一使用字符串形式的类别ID
            else:
                class_name = "0"  # 默认类别
                
            all_samples.append({
                'img_name': img_name,
                'score': img_mus_score,
                'class_name': class_name
            })
        
        # 2. 按MUS排序
        sorted_samples = sorted(all_samples, key=lambda x: x['score'], reverse=(strategy=='max'))
        
        # 3. 提取CDB参数（从传入的labeled_counts中获取）
        # 将类别统计转换为ID -> count的映射
        labeled_counts_by_id = {}
        for cls_name, stats in labeled_counts.items():
            try:
                # 假设stats中有id字段
                if isinstance(stats, dict) and 'id' in stats:
                    cls_id = int(stats['id'])
                    labeled_counts_by_id[cls_id] = stats.get("labeled", 0)
                else:
                    # 尝试直接使用类别名称作为ID
                    cls_id = int(cls_name)
                    labeled_counts_by_id[cls_id] = stats.get("labeled", 0) if isinstance(stats, dict) else 0
            except (ValueError, TypeError):
                # 如果无法转换，跳过
                continue
        
        # 4. 计算CDB预算
        current_labels = [s['class_name'] for s in sorted_samples]
        cdb_quota = self.calculate_cdb_budget(
            labeled_counts=labeled_counts_by_id,
            current_labels=current_labels,
            total_budget=total_budget
        )
        
        # 5. 按预算筛选，但确保达到total_budget数量
        selected = []
        remaining = {cls: cnt for cls, cnt in cdb_quota.items()}
        
        # 第一轮：按照CDB预算选择
        for sample in sorted_samples:
            if remaining.get(sample['class_name'], 0) > 0:
                selected.append(sample['img_name'])
                remaining[sample['class_name']] -= 1
            
            # 如果已达到目标数量，跳出
            if len(selected) >= total_budget:
                break
        
        # 第二轮：如果还没达到目标数量，按MUS得分继续选择
        if len(selected) < total_budget:
            remaining_samples = [s for s in sorted_samples 
                               if s['img_name'] not in selected]
            for sample in remaining_samples:
                selected.append(sample['img_name'])
                if len(selected) >= total_budget:
                    break
        
        return selected[:total_budget]  # 确保不超过预算

    def select_samples(self, unlabeled_results: Dict, num_samples: int) -> List[str]:
        """兼容接口：简单的MUS采样，用于回退模式"""
        all_samples = []
        for img_name, info in unlabeled_results.items():
            if 'uncertainty' in info and 'mus' in info['uncertainty']:
                mus_scores = info['uncertainty']['mus']
                if isinstance(mus_scores, dict):
                    for bbox_id, score in mus_scores.items():
                        all_samples.append((img_name, score))
                else:
                    # 如果是单个值，使用图像名称
                    all_samples.append((img_name, float(mus_scores)))
        
        # 按MUS得分排序并选择前num_samples个
        sorted_samples = sorted(all_samples, key=lambda x: x[1], reverse=True)
        return [sample[0] for sample in sorted_samples[:num_samples]]

    def get_selection_report(self, selected_samples: List[str], unlabeled_results: Dict) -> Dict:
        """生成选择报告"""
        report = {
            'strategy': 'MUS-CDB',
            'selected_count': len(selected_samples),
            'total_available': len(unlabeled_results),
            'selection_ratio': len(selected_samples) / len(unlabeled_results) if unlabeled_results else 0,
            'selected_samples': selected_samples,
            'metrics': {
                'average_mus_score': 0.0,
                'score_distribution': {}
            }
        }
        
        # 计算选中样本的平均MUS得分
        if selected_samples and unlabeled_results:
            mus_scores = []
            for sample_name in selected_samples:
                if sample_name in unlabeled_results:
                    info = unlabeled_results[sample_name]
                    if 'uncertainty' in info and 'mus' in info['uncertainty']:
                        mus_score = info['uncertainty']['mus']
                        if isinstance(mus_score, dict):
                            # 如果是字典，取平均值
                            mus_scores.append(np.mean(list(mus_score.values())))
                        else:
                            mus_scores.append(float(mus_score))
            
            if mus_scores:
                report['metrics']['average_mus_score'] = np.mean(mus_scores)
                report['metrics']['score_distribution'] = {
                    'min': float(np.min(mus_scores)),
                    'max': float(np.max(mus_scores)),
                    'std': float(np.std(mus_scores))
                }
        
        return report