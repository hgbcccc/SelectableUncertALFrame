score_threshold = 0.08
max_boxes_per_img = 250
nms_iou_threshold = 0.4

active_learning = dict(
    ann_file=
    'data/ForestDamages/active_learning_cascade_ssc/annotations/instances_unlabeled.json',
    data_prefix=dict(
        img='data/ForestDamages/active_learning_cascade_ssc/images_unlabeled'),
    data_root='data/ForestDamages/active_learning_cascade_ssc',
    inference_options=dict(
        batch_size=16,
        sample_size=0,
        save_results=False,
        score_thr=score_threshold,
        selected_metric='ssc_score',
        uncertainty_methods=[
            'ssc',
        ]),
    max_iterations=16,
    sample_selection=dict(
        num_samples=200,
        rl_metric='',
        sample_selector='default',
        uncertainty_metric='ssc_score'),
    train_pool_cfg=dict(
        ann_file='annotations/instances_labeled_train.json',
        data_prefix=dict(img='images_labeled_train'),
        data_root='data/ForestDamages/active_learning_cascade_ssc'))
auto_scale_lr = dict(base_batch_size=4, enable=False)
backend_args = None
data_root = 'data/ForestDamages/active_learning_cascade_ssc'
dataset_type = 'ForestDamagesDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
interval = 1
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)

metainfo = dict(
    classes=(
        'Aspen',
        'Birch',
        'Other',
        'Pine',
        'Spruce',
    ),
    palette=[
        (
            220,
            20,
            60,
        ),
        (
            119,
            11,
            32,
        ),
        (
            0,
            0,
            142,
        ),
        (
            0,
            0,
            230,
        ),
        (
            106,
            0,
            228,
        ),
    ])
model = dict(
    backbone=dict(
        depth=101,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet101', type='Pretrained'),
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
        type='ResNet'),
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
        bbox_head=[
            dict(
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
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=1.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                num_classes=5,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.05,
                        0.05,
                        0.1,
                        0.1,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=1.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                num_classes=5,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.033,
                        0.033,
                        0.067,
                        0.067,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=1.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                num_classes=5,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
        ],
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
        num_stages=3,
        stage_loss_weights=[
            1,
            0.5,
            0.25,
        ],
        type='CascadeRoIHead'),
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
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=max_boxes_per_img,
            nms=dict(iou_threshold=nms_iou_threshold, type='nms'),
            score_thr=score_threshold),
        rpn=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.6, type='nms'),
            nms_pre=2000)),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.6,
                    neg_iou_thr=0.6,
                    pos_iou_thr=0.6,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.7,
                    neg_iou_thr=0.7,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
        ],
        rpn=dict(
            allowed_border=0,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
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
            nms_pre=2000)),
    type='CascadeRCNN')
nms_iou_threshold = 0.5
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        T_max=3,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=3,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
]
resume = False

test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='annotations/instances_val2024.json',
        backend_args=None,
        data_prefix=dict(img='val2024/'),
        data_root='data/ForestDamages',
        metainfo=dict(
            classes=(
                'Aspen',
                'Birch',
                'Other',
                'Pine',
                'Spruce',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    142,
                ),
                (
                    0,
                    0,
                    230,
                ),
                (
                    106,
                    0,
                    228,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1280,
                1280,
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
    num_workers=4,
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
train_cfg = dict(max_epochs=3, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file=
        'data/ForestDamages/active_learning_cascade_ssc/annotations/instances_labeled_train.json',
        backend_args=None,
        data_prefix=dict(
            img=
            'data/ForestDamages/active_learning_cascade_ssc/images_labeled_train'
        ),
        data_root='data/ForestDamages/active_learning_cascade_ssc',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(
            classes=(
                'Aspen',
                'Birch',
                'Other',
                'Pine',
                'Spruce',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    142,
                ),
                (
                    0,
                    0,
                    230,
                ),
                (
                    106,
                    0,
                    228,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1280,
                1280,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='ActiveCocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1280,
        1280,
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
        -20,
        20,
    ), type='RandomRotate'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        'data/ForestDamages/active_learning_cascade_ssc/annotations/instances_labeled_val.json',
        backend_args=None,
        data_prefix=dict(
            img=
            'data/ForestDamages/active_learning_cascade_ssc/images_labeled_val'
        ),
        data_root='data/ForestDamages/active_learning_cascade_ssc',
        metainfo=dict(
            classes=(
                'Aspen',
                'Birch',
                'Other',
                'Pine',
                'Spruce',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    142,
                ),
                (
                    0,
                    0,
                    230,
                ),
                (
                    106,
                    0,
                    228,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1536,
                1536,
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
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    'data/ForestDamages/active_learning_cascade_ssc/annotations/instances_labeled_val.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = 'new_work_dirs/cascade-rcnn_ssc_16_200'
