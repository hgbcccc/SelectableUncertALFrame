import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import List
from pathlib import Path  # 正确导入Path模块
from sual.core.datasets import ActiveCocoDataset

def _calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """计算两个框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)

def _get_box_centers(boxes: np.ndarray) -> np.ndarray:
    """计算所有框的中心点坐标"""
    centers = np.zeros((len(boxes), 2))
    centers[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x center
    centers[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y center
    return centers

def _calculate_center_entropy(centers: np.ndarray, grid_size: int = 8) -> float:
    """计算中心点分布的熵"""
    x_bins = np.linspace(0, 1, grid_size + 1)
    y_bins = np.linspace(0, 1, grid_size + 1)
    
    hist, _, _ = np.histogram2d(centers[:, 0], centers[:, 1], 
                                bins=[x_bins, y_bins])
    hist = hist.flatten()
    prob = hist / (len(centers) + 1e-6)
    prob = prob[prob > 0]
    
    return -np.sum(prob * np.log2(prob + 1e-6))

def _calculate_center_variance(centers: np.ndarray) -> float:
    """计算中心点坐标的方差"""
    center_var_x = np.var(centers[:, 0])
    center_var_y = np.var(centers[:, 1])
    return (center_var_x + center_var_y) / 2

def _calculate_sor(box: np.ndarray, all_boxes: np.ndarray) -> float:
    """计算单个框的空间遮挡率(SOR)"""
    x1, y1, x2, y2 = box
    overlap_area = 0

    for other_box in all_boxes:
        if np.array_equal(box, other_box):
            continue
        x1_other, y1_other, x2_other, y2_other = other_box

        # 计算重叠区域
        inter_x1 = max(x1, x1_other)
        inter_y1 = max(y1, y1_other)
        inter_x2 = min(x2, x2_other)
        inter_y2 = min(y2, y2_other)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:  # 有重叠
            overlap_width = inter_x2 - inter_x1
            overlap_height = inter_y2 - inter_y1
            overlap_area += overlap_width * overlap_height

    total_area = (x2 - x1) * (y2 - y1)  # 计算框的总面积
    return overlap_area / (total_area + 1e-6)  # 返回 SOR 值
