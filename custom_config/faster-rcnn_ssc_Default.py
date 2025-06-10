score_threshold = 0.08
max_boxes_per_img = 250
nms_iou_threshold = 0.4


active_learning = dict(
    data_root='data/ForestDamages/active_learning_default',
    ann_file='data/ForestDamages/active_learning_default/annotations/instances_unlabeled.json',
    data_prefix=dict(img='data/ForestDamages/active_learning_default/images_unlabeled'),
    # 训练池的SSC计算配置
    train_pool_cfg = dict(
        data_root='data/ForestDamages/active_learning_default',
        ann_file='annotations/instances_labeled_train.json',
        data_prefix=dict(
            img='images_labeled_train'
        )
    ),
    inference_options=dict(
        score_thr=score_threshold ,
        uncertainty_methods=['ssc'],
        selected_metric="ssc_score",
        sample_size=0,
        batch_size=16
    ),
    max_iterations=16,
    sample_selection=dict(
        num_samples=200,
        uncertainty_metric='ssc_score',
        sample_selector="default" , # 'Wasserstein' 或 'RL' 还有默认的ssc_score最大值排序选择
        rl_metric='' # 强化学习中的度量方法可选: 'wasserstein', 'kl', 'js', 'euclidean', 'simple_stats', 'percentile', 'ensemble'
    )
)


auto_scale_lr = dict(base_batch_size=4, enable=False)
backend_args = None
data_root = 'data/ForestDamages/active_learning_default'
dataset_type = 'ForestDamagesDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts= 3,
        type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
interval = 1
load_from = None
# log_level = 'WARNING'
# log_level = 'ERROR'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs=3  # 每轮主动学习内的训练轮数增加到3轮

metainfo = dict(
    classes=('Aspen', 'Birch', 'Other', 'Pine', 'Spruce'), 
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228)]
)

model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://resnext101_32x4d'),
        norm_eval=True),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            # 修改损失函数为Focal Loss处理类别不平衡
            loss_cls=dict(
                loss_weight=1.0, 
                type='FocalLoss',  # 使用Focal Loss替代CrossEntropyLoss
                use_sigmoid=True,  # 启用sigmoid
                gamma=2.0,  # 聚焦参数
                alpha=0.25  # 平衡因子
            ),
            num_classes=5,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        # 同样修改RPN的分类损失函数
        loss_cls=dict(
            loss_weight=1.0, 
            type='FocalLoss', 
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25
        ),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=max_boxes_per_img,  # 增加以获取更多候选框
            nms=dict(iou_threshold=nms_iou_threshold, type='nms'),  # 放宽NMS阈值
            score_thr=score_threshold),  # 降低得分阈值
        rpn=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.6, type='nms'),
            nms_pre=2000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                # 改回RandomSampler，避免OHEM采样器的问题
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,  # 保持正样本比例
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0))
)
optim_wrapper = dict(
    # 使用AdamW优化器，通常在类别不平衡情况下表现更好
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 降低学习率
        weight_decay=0.05,  # 增加权重衰减
        betas=(0.9, 0.999)
    ),
    type='OptimWrapper',
    # 添加梯度裁剪，防止梯度爆炸
    clip_grad=dict(max_norm=35, norm_type=2)
)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001,
        type='LinearLR'),  # 更温和的预热
    dict(
        by_epoch=True, 
        # 余弦退火学习率调度
        type='CosineAnnealingLR',
        begin=0,
        T_max=3,  # 每轮主动学习内的训练轮数
        end=3,
        eta_min=1e-6,
        convert_to_iter_based=True
    )
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,  # 减小批量大小，避免OOM
    dataset=dict(
        ann_file='annotations/instances_val2024.json',
        backend_args=None,
        data_prefix=dict(img='val2024/'),
        data_root='data/ForestDamages',
        metainfo=metainfo,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1280,
                1280,
            ), type='Resize'),  # 修改为更高分辨率
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='ForestDamagesDataset'),
    drop_last=False,
    num_workers=4,  # 减少工作线程数
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/ForestDamages/annotations/instances_val2024.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1536,
        1536,
    ), type='Resize'),  # 修改为更高分辨率
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=3, type='EpochBasedTrainLoop', val_interval=1)  # 增加到3轮
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
    ann_file='data/ForestDamages/active_learning_default/annotations/instances_labeled_train.json',
        backend_args=None,
        data_prefix=dict(img='data/ForestDamages/active_learning_default/images_labeled_train'),
        data_root='data/ForestDamages/active_learning_default',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=metainfo,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1280,
                1280,
            ), type='Resize'),  # 修改为更高分辨率
            dict(prob=0.5, type='RandomFlip'),
            # # 增强数据增强，改善类别不平衡
            # dict(
            #     brightness_delta=32,
            #     contrast_range=(0.5, 1.5),
            #     hue_delta=18,
            #     saturation_range=(0.5, 1.5),
            #     type='PhotoMetricDistortion'
            # ),
            # dict(
            #     min_crop_size=0.3,
            #     min_ious=(0.4, 0.5, 0.6, 0.7),
            #     type='MinIoURandomCrop'
            # ),
            # dict(angle_range=(
            #     -20,
            #     20,
            # ), type='RandomRotate'),
            dict(type='PackDetInputs'),
        ],
        # 添加类别平衡采样器
        type='ActiveCocoDataset'),
    num_workers=4,  # 减少工作线程数，节约资源
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler')  # 使用DefaultSampler    sampler=dict(shuffle=Tru
)

train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1280,
        1280,
    ), type='Resize'),  # 修改为更高分辨率
    dict(
        brightness_delta=32,
        contrast_range=(
            0.5,
            1.5,
        ),
        hue_delta=18,
        saturation_range=(
            0.5,
            1.5,
        ),
        type='PhotoMetricDistortion'),
    dict(
        min_crop_size=0.3,
        min_ious=(
            0.4,
            0.5,
            0.6,
            0.7,
        ),
        type='MinIoURandomCrop'),
    dict(angle_range=(
        -20,
        20,
    ), type='RandomRotate'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,  # 减小批量大小，适应更大的图像尺寸
    dataset=dict(
        ann_file='data/ForestDamages/active_learning_default/annotations/instances_labeled_val.json',
        backend_args=None,
        data_prefix=dict(img='data/ForestDamages/active_learning_default/images_labeled_val'),
        data_root='data/ForestDamages/active_learning_default',
        metainfo=metainfo,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1536,
                1536,
            ), type='Resize'),  # 修改为更高分辨率
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='ActiveCocoDataset'),
    drop_last=False,
    num_workers=4,  # 减少工作线程数
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/ForestDamages/active_learning_default/annotations/instances_labeled_val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    # 增加每类别指标，更好地评估类别不平衡的表现
    classwise=True,
    type='CocoMetric')

# 修改vis_backends部分
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')  # 保留TensorBoard后端
]

# visualizer部分也需要相应更新
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')  # 保留TensorBoard后端
    ])
work_dir = 'work_dirs/faster-rcnn_ssc_default_16_200'  # 修改工作目录