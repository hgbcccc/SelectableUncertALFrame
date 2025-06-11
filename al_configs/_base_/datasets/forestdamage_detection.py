dataset_type = 'ForestDamagesDataset'
data_root = 'data/ForestDamages/active_learning_cascade_sor'
metainfo = dict(
    classes=('Aspen', 'Birch', 'Other', 'Pine', 'Spruce'), 
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228)]
)



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
        ann_file='data/ForestDamages/active_learning_cascade_sor/annotations/instances_labeled_train.json',
        backend_args=None,
        data_prefix=dict(img='data/ForestDamages/active_learning_cascade_sor/images_labeled_train'),
        data_root='data/ForestDamages/active_learning_cascade_sor',
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
        ann_file='data/ForestDamages/active_learning_cascade_sor/annotations/instances_labeled_val.json',
        backend_args=None,
        data_prefix=dict(img='data/ForestDamages/active_learning_cascade_sor/images_labeled_val'),
        data_root='data/ForestDamages/active_learning_cascade_sor',
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
    ann_file='data/ForestDamages/active_learning_cascade_sor/annotations/instances_labeled_val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    # 增加每类别指标，更好地评估类别不平衡的表现
    classwise=True,
    type='CocoMetric')