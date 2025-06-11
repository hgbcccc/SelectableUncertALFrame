# SSC (Spatial and Scene Complexity) 采样策略配置
# 基于空间和场景复杂度的主动学习采样

active_learning = dict(
    ann_file=
    'data/ForestDamages/active_learning_cascade_ssc/annotations/instances_unlabeled.json',
    data_prefix=dict(
        img='data/ForestDamages/active_learning_cascade_ssc/images_unlabeled'),
    data_root='data/ForestDamages/active_learning_cascade_ssc',
    inference_options=dict(
        batch_size=16,
        sample_size=0,
        save_results=True,
        score_thr=0.08,
        selected_metric='ssc_score',
        uncertainty_methods=[
            'ssc',
        ]),
    max_iterations=16,
    sample_selection=dict(
        num_samples=200,
        rl_metric='',
        sample_selector='default',
        uncertainty_metric='ssc_score'
    ),
    train_pool_cfg=dict(
        ann_file='annotations/instances_labeled_train.json',
        data_prefix=dict(img='images_labeled_train'),
        data_root='data/ForestDamages/active_learning_cascade_ssc'
    )
)