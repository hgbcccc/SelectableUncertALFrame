
# import cv2
# import torch
# import torch.nn as nn
# from typing import List, Dict, Optional
# import numpy as np
# from pathlib import Path  # 正确导入Path模块
# # 在此处导入避免循环导入
# # from sual.core.datasets.activate_datasets import ActiveCocoDataset
# # from sual.core.uncertainty.measures.utils import  FeatureExtractor, FeatureExtractorFactory

# class CoreSetSelector:
#     """Core-set样本选择模块"""
#     def __init__(self, feature_extractor: FeatureExtractor):
#         #正确的导入位置，在使用时候进行导入

        
#         self.feature_extractor = feature_extractor
#         self.feature_cache = {}
    
#     def _get_image_path(self, dataset: ActiveCocoDataset, img_name: str) -> str:

#         """获取完整图像路径"""
#         return str(dataset.img_unlabeled / img_name)
    
#     def _compute_features(self, dataset: ActiveCocoDataset, img_names: List[str]) -> np.ndarray:
#         """批量计算特征"""
#         features = []
#         for name in img_names:
#             if name not in self.feature_cache:
#                 img_path = self._get_image_path(dataset, name)
#                 self.feature_cache[name] = self.feature_extractor.extract(img_path)
#             features.append(self.feature_cache[name].cpu().numpy())
#         return np.array(features)
    
#     def select(self, dataset: ActiveCocoDataset, candidates: List[str], budget: int) -> List[str]:
#         """执行Core-set选择"""
#         if len(candidates) <= budget:
#             return candidates.copy()
        
#         # 计算特征
#         features = self._compute_features(dataset, candidates)
#         features = np.nan_to_num(features)   # 可能有问题
#         features = np.clip(features, -1e5, 1e5)  # 防止溢出   # 可能有问题
#         features = features / np.linalg.norm(features, axis=1, keepdims=True)
        
#         # 贪婪k-center算法
#         selected = []
#         remaining_indices = list(range(len(candidates)))
        
#         # 初始化第一个样本
#         first_idx = np.random.choice(len(candidates))
#         selected.append(first_idx)
#         remaining_indices.remove(first_idx)
        
#         for _ in range(budget-1):
#             # 计算最小距离
#             dists = np.linalg.norm(
#                 features[remaining_indices] - features[selected][:, None], 
#                 axis=2
#             )
#             min_dists = np.min(dists, axis=0)
            
#             # 选择最远点
#             farthest = np.argmax(min_dists)
#             selected_idx = remaining_indices[farthest]
            
#             selected.append(selected_idx)
#             remaining_indices.pop(farthest)
        
#         return [candidates[i] for i in selected]