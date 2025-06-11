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
        T_max=48,  # 每轮主动学习内的训练轮数
        end=48,
        eta_min=1e-6,
        convert_to_iter_based=True
    )
]