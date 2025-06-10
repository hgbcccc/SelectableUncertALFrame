score_threshold = 0.08
max_boxes_per_img = 250
nms_iou_threshold = 0.4
data_root = 'data/ForestDamages/active_learning_detr_ssc'
dataset_type = 'ForestDamagesDataset'
metainfo = dict(
    classes=('Aspen', 'Birch', 'Other', 'Pine', 'Spruce'), 
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228)]
)

max_epochs = 3 


active_learning = dict(
    data_root='data/ForestDamages/active_learning_detr_ssc',
    ann_file='data/ForestDamages/active_learning_detr_ssc/annotations/instances_unlabeled.json',
    data_prefix=dict(img='data/ForestDamages/active_learning_detr_ssc/images_unlabeled'),
    # 训练池的SSC计算配置
    train_pool_cfg = dict(
        data_root='data/ForestDamages/active_learning_detr_ssc',
        ann_file='annotations/instances_labeled_train.json',
        data_prefix=dict(
            img='images_labeled_train'
        )
    ),
    inference_options=dict(
        save_results=True,
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
data_root = 'data/ForestDamages/active_learning_detr_ssc'
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
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)

metainfo = dict(
    classes=('Aspen', 'Birch', 'Other', 'Pine', 'Spruce'), 
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228)]
)

max_epochs = 3  # 修改为与Cascade-RCNN一致的3个epoch

model = dict(
    backbone=dict(
        depth=101,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet101', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        embed_dims=256,
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25
        ),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=5,  # 修改为5个类别
        type='DETRHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=1,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(
                batch_first=True, dropout=0.1, embed_dims=256, num_heads=8),
            ffn_cfg=dict(
                act_cfg=dict(inplace=True, type='ReLU'),
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.1,
                num_fcs=2),
            self_attn_cfg=dict(
                batch_first=True, dropout=0.1, embed_dims=256, num_heads=8)),
        num_layers=6,
        return_intermediate=True),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                act_cfg=dict(inplace=True, type='ReLU'),
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.1,
                num_fcs=2),
            self_attn_cfg=dict(
                batch_first=True, dropout=0.1, embed_dims=256, num_heads=8)),
        num_layers=6),
    neck=dict(
        act_cfg=None,
        in_channels=[
            2048,
        ],
        kernel_size=1,
        norm_cfg=None,
        num_outs=1,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=100,
    positional_encoding=dict(normalize=True, num_feats=128),
    test_cfg=dict(max_per_img=max_boxes_per_img),  # 使用相同的max_boxes_per_img
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='ClassificationCost', weight=1.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='DETR')
# optim_wrapper = dict(
#     optimizer=dict(
#         type='AdamW',
#         lr=0.0001,
#         weight_decay=0.05,
#         betas=(0.9, 0.999)
#     ),
#     type='OptimWrapper',
#     clip_grad=dict(max_norm=35, norm_type=2),
#     paramwise_cfg=dict(
#         custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0)))
# )

optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(decay_mult=1.0, lr_mult=0.1))),
    type='OptimWrapper')
# param_scheduler = [
#     dict(
#         begin=0, by_epoch=False, end=500, start_factor=0.001,
#         type='LinearLR'),
#     dict(
#         by_epoch=True, 
#         type='CosineAnnealingLR',
#         begin=0,
#         T_max=3,  # 每轮主动学习内的训练轮数
#         end=3,
#         eta_min=1e-6,
#         convert_to_iter_based=True
#     )
# ]
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=48,
        gamma=0.1,
        milestones=[
            33,
        ],
        type='MultiStepLR'),
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
        metainfo=metainfo,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(1280, 1280), type='Resize'),
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
    classwise=True,
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(1536, 1536), type='Resize'),
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
        ann_file='data/ForestDamages/active_learning_detr_ssc/annotations/instances_labeled_train.json',
        backend_args=None,
        data_prefix=dict(img='data/ForestDamages/active_learning_detr_ssc/images_labeled_train'),
        data_root='data/ForestDamages/active_learning_detr_ssc',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=metainfo,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(1280, 1280), type='Resize'),
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
    dict(keep_ratio=True, scale=(1280, 1280), type='Resize'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            400,
                            1333,
                        ),
                        (
                            500,
                            1333,
                        ),
                        (
                            600,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type='absolute_range',
                    type='RandomCrop'),
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='data/ForestDamages/active_learning_detr_ssc/annotations/instances_labeled_val.json',
        backend_args=None,
        data_prefix=dict(img='data/ForestDamages/active_learning_detr_ssc/images_labeled_val'),
        data_root='data/ForestDamages/active_learning_detr_ssc',
        metainfo=metainfo,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(1536, 1536), type='Resize'),
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
    ann_file='data/ForestDamages/active_learning_detr_ssc/annotations/instances_labeled_val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    classwise=True,
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ])

work_dir = 'work_dirs/detr_ssc_16_200'
