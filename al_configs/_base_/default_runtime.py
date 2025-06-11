default_scope = 'mmdet'

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
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)


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
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
