score_threshold = 0.08
max_boxes_per_img = 300
nms_iou_threshold = 0.6


active_learning = dict(
    data_root='data/ForestDamages/active_learning_wasserstein',
    ann_file='data/ForestDamages/active_learning_wasserstein/annotations/instances_unlabeled.json',
    data_prefix=dict(img='data/ForestDamages/active_learning_wasserstein/images_unlabeled'),
    # 训练池的SSC计算配置
    train_pool_cfg = dict(
        data_root='data/ForestDamages/active_learning_wasserstein',
        ann_file='annotations/instances_labeled_train.json',
        data_prefix=dict(
            img='images_labeled_train'
        )
    ),
    inference_options=dict(
        score_thr=score_threshold ,
        uncertainty_methods=['ssc'],
        selected_metric="ssc_score",
        sample_size=0
    ),
    max_iterations=16,
    sample_selection=dict(
        num_samples=100,
        uncertainty_metric='ssc_score',
        sample_selector="Wasserstein" , # 'Wasserstein' 或 'RL' 还有默认的ssc_score最大值排序选择
        rl_metric='' # 强化学习中的度量方法可选: 'wasserstein', 'kl', 'js', 'euclidean', 'simple_stats', 'percentile', 'ensemble'
    )
)


auto_scale_lr = dict(base_batch_size=4, enable=False)
backend_args = None
data_root = 'data/ForestDamages/active_learning_wasserstein'
dataset_type = 'ForestDamagesDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
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
max_epochs=1

metainfo = dict(
    classes=('Aspen', 'Birch', 'Other', 'Pine', 'Spruce'), 
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228)]
)

model = dict(
    backbone=dict(
        base_width=4,
        depth=101,
        frozen_stages=1,
        groups=32,
        init_cfg=dict(
            checkpoint='open-mmlab://resnext101_32x4d', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNeXt'),
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
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
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
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
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
                ignore_iof_thr=-1,
                match_low_quality=True,  # 允许低质量匹配
                min_pos_iou=0.3,  # 降低正样本IoU阈值
                neg_iou_thr=0.3,  # 降低负样本IoU阈值
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.3,  # 增加正样本比例
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.2,  # 降低最小正样本IoU
                neg_iou_thr=0.2,  # 降低负样本IoU阈值
                pos_iou_thr=0.5,  # 降低正样本IoU阈值
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=3000)),
    type='FasterRCNN')
optim_wrapper = dict(
    optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=20000, start_factor=0.01,
        type='LinearLR'),
    dict(
        by_epoch=True, gamma=0.1, milestones=[
            350,
            450,
        ], type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='annotations/instances_val2024.json',
        backend_args=None,
        data_prefix=dict(img='val2024/'),
        data_root='data/ForestDamages',
        metainfo=metainfo,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
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
    num_workers=8,
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
        1024,
        1024,
    ), type='Resize'),
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
train_cfg = dict(max_epochs=1, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
    ann_file='data/ForestDamages/active_learning_wasserstein/annotations/instances_labeled_train.json',
        backend_args=None,
        data_prefix=dict(img='data/ForestDamages/active_learning_wasserstein/images_labeled_train'),
        data_root='data/ForestDamages/active_learning_wasserstein',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=metainfo,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='ActiveCocoDataset'),
    num_workers=12,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
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
        -10,
        10,
    ), type='RandomRotate'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='data/ForestDamages/active_learning_wasserstein/annotations/instances_labeled_val.json',
        backend_args=None,
        data_prefix=dict(img='data/ForestDamages/active_learning_wasserstein/images_labeled_val'),
        data_root='data/ForestDamages/active_learning_wasserstein',
        metainfo=metainfo,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
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
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/ForestDamages/active_learning_wasserstein/annotations/instances_labeled_val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
# vis_backends = [
#     dict(type='LocalVisBackend'),
# ]
# visualizer = dict(
#     name='visualizer',
#     type='DetLocalVisualizer',
#     vis_backends=[
#         dict(type='LocalVisBackend'),
#     ])

# 修改vis_backends部分
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')  # 添加TensorBoard后端
]

# visualizer部分也需要相应更新
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')  # 添加TensorBoard后端
    ])
work_dir = 'al_laboratry'