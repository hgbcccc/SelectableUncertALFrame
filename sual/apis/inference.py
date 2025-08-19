# sual/apis/inference.py
import copy
import torch
import numpy as np
from typing import Optional, Sequence, Union, Dict, List
import types
import torch.nn.functional as F
from mmcv.transforms import Compose
from mmcv.ops import RoIPool
import torch.nn as nn
from mmdet.utils import get_test_pipeline_cfg
from mmdet.structures import DetDataSample, SampleList
from mmdet.apis import inference_detector

def sual_inference_detector(
    model: nn.Module,
    imgs: Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]],
    test_pipeline: Optional[Compose] = None,
    text_prompt: Optional[str] = None,
    custom_entities: bool = False,
) -> Union[DetDataSample, SampleList]:
    """增强版推理函数，准确获取每个检测框的完整类别概率分布。
    
    专门针对Faster R-CNN等两阶段检测器优化，获取NMS前后的精确映射关系。
    
    Args:
        model (nn.Module): 检测模型
        imgs: 图像或图像路径
        test_pipeline: 测试数据管道
        text_prompt: 文本提示
        custom_entities: 是否使用自定义实体
        
    Returns:
        DetDataSample 或 SampleList: 增强的检测结果，包含all_cls_probs属性
    """
    # 保存原始的multiclass_nms函数
    from mmdet.models.utils.misc import multiclass_nms
    original_multiclass_nms = multiclass_nms
    
    # 创建一个全局字典用于存储NMS前后的映射关系
    nms_mapping = {}
    
    # 增强版的NMS函数，追踪索引映射关系
    def enhanced_multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None, return_inds=False, **kwargs):
        """增强版的NMS函数，记录映射关系"""
        # 生成唯一ID用于识别当前批次
        batch_id = id(multi_scores)
        
        # 保存原始概率分布
        original_scores = multi_scores.clone()
        
        # 调用原始NMS函数
        if return_inds:
            det_bboxes, det_labels, inds = original_multiclass_nms(
                multi_bboxes, multi_scores, score_thr, nms_cfg, max_num, 
                score_factors, return_inds=True, **kwargs
            )
            # 记录映射关系
            nms_mapping[batch_id] = {
                'original_scores': original_scores,
                'det_inds': inds,
                'det_labels': det_labels
            }
            return det_bboxes, det_labels, inds
        else:
            det_bboxes, det_labels = original_multiclass_nms(
                multi_bboxes, multi_scores, score_thr, nms_cfg, max_num, 
                score_factors, return_inds=False, **kwargs
            )
            # 记录映射关系
            nms_mapping[batch_id] = {
                'original_scores': original_scores,
                'det_labels': det_labels
            }
            return det_bboxes, det_labels
    
    # 保存原始的_predict_by_feat_single方法
    if hasattr(model, 'roi_head') and hasattr(model.roi_head, 'bbox_head'):
        original_predict = model.roi_head.bbox_head._predict_by_feat_single
        
        # 增强版的预测方法
        def enhanced_predict(self, roi, cls_score, bbox_pred, img_meta, rescale=False, rcnn_test_cfg=None):
            """增强版预测方法，保存并关联原始类别概率"""
            # 替换NMS函数
            from mmdet.models.utils import misc
            misc.multiclass_nms = enhanced_multiclass_nms
            
            # 调用原始方法
            results = original_predict(self, roi, cls_score, bbox_pred, img_meta, rescale, rcnn_test_cfg)
            
            # 恢复原始NMS函数
            misc.multiclass_nms = original_multiclass_nms
            
            # 处理映射关系
            batch_id = id(cls_score)
            if batch_id in nms_mapping:
                mapping_info = nms_mapping[batch_id]
                original_scores = mapping_info['original_scores']
                det_labels = mapping_info['det_labels']
                
                # 获取完整的类别概率
                if cls_score is not None:
                    # 将分类分数转换为概率
                    if self.custom_cls_channels:
                        all_probs = self.loss_cls.get_activation(cls_score)
                    else:
                        all_probs = F.softmax(cls_score, dim=-1)
                    
                    # 为每个保留的框匹配原始概率
                    num_instances = len(results.scores)
                    num_classes = all_probs.shape[1]
                    matched_probs = torch.zeros((num_instances, num_classes), device=results.scores.device)
                    
                    # 使用NMS返回的索引获取精确映射
                    if 'det_inds' in mapping_info:
                        det_inds = mapping_info['det_inds']
                        for i, (idx, label) in enumerate(zip(det_inds, det_labels)):
                            # 获取原始概率分布
                            matched_probs[i] = all_probs[idx]
                    else:
                        # 如果没有索引信息，则尝试用分数匹配
                        for i, (label, score) in enumerate(zip(det_labels, results.scores)):
                            # 在允许的误差范围内查找匹配的分数
                            cls_scores = original_scores[:, label]
                            matched = torch.isclose(cls_scores, score, atol=1e-5)
                            if matched.any():
                                idx = torch.where(matched)[0][0].item()
                                matched_probs[i] = all_probs[idx]
                    
                    # 保存匹配的概率
                    results.all_cls_probs = matched_probs
                
                # 清理不再需要的映射信息
                del nms_mapping[batch_id]
            
            return results
        
        # 应用增强方法
        model.roi_head.bbox_head._predict_by_feat_single = types.MethodType(enhanced_predict, model.roi_head.bbox_head)
    
    # 对于YOLO等单阶段检测器也可以增加类似的处理
    elif hasattr(model, 'bbox_head'):
        # 单阶段检测器的处理逻辑...
        pass
    
    try:
        # 使用标准推理函数
        results = inference_detector(model, imgs, test_pipeline, text_prompt, custom_entities)
        return results
    finally:
        # 恢复原始方法
        if hasattr(model, 'roi_head') and hasattr(model.roi_head, 'bbox_head'):
            model.roi_head.bbox_head._predict_by_feat_single = original_predict
        # 清理所有映射信息
        nms_mapping.clear()