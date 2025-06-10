import numpy as np
from typing import Dict
from mmdet.structures import DetDataSample

def compute_value_metrics(result: DetDataSample) -> Dict[str, float]:
        """计算图片价值的三个指标：难度、信息量和多样性
        
        Args:
            result (DetDataSample): MMDetection的检测结果
            
        Returns:
            Dict[str, float]: 包含三个指标和综合得分的字典
        """
        # 获取所有检测框的信息
        scores = result.pred_instances.scores.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()
        
        # 检查是否有检测结果
        if len(scores) == 0:
            return {
                'difficulty_score': 0.0,
                'information_score': 0.0,
                'diversity_score': 0.0,
                'total_value_score': 0.0,
                'num_objects': 0,
                'num_classes': 0,
                'class_distribution': {}
            }
        
        # 1. 难度指标 (基于熵)
        norm_scores = scores / (np.sum(scores) + 1e-10)
        difficulty = -np.sum(norm_scores * np.log(norm_scores + 1e-10))
        difficulty = difficulty / (np.log(len(scores) + 1e-10))  # 归一化熵
        
        # 2. 信息指标 (基于所有检测框的置信度)
        information = np.mean(scores)
        
        # 3. 多样性指标 (基于类别数量)
        unique_classes = np.unique(labels)
        num_classes = len(unique_classes)
        diversity = num_classes / 80.0  # 假设COCO数据集80类，归一化
        
        # 4. 综合得分 (三个指标的加权和)
        total_score = (difficulty + information + diversity) / 3.0
        
        # 5. 统计类别分布
        class_distribution = {
            int(label): int(np.sum(labels == label)) 
            for label in unique_classes
        }
        
        return {
            'difficulty_score': float(difficulty),
            'information_score': float(information),
            'diversity_score': float(diversity),
            'total_value_score': float(total_score),
            'num_objects': int(len(scores)),
            'num_classes': int(num_classes),
            'class_distribution': class_distribution
        }