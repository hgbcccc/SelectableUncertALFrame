score_threshold = 0.08
max_boxes_per_img = 250
nms_iou_threshold = 0.4

_base_ = [
    '../_base_/al_sampling/least_confidence.py', # 采样策略
    '../_base_/default_runtime.py', # runtime
    '../_base_/models/cascade-rcnn_r101.py', # 模型
    '../_base_/schedules/custom_schedule.py', # 学习率策略
    '../_base_/datasets/forestdamage_detection.py', # 数据集
] 

