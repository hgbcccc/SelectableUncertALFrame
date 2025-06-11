active_learning = dict(
    data_root='data/ForestDamages/active_learning_cascade_random',
    ann_file='data/ForestDamages/active_learning_cascade_random/annotations/instances_unlabeled.json',
    data_prefix=dict(img='data/ForestDamages/active_learning_cascade_random/images_unlabeled'),
    # 训练池的SSC计算配置
    train_pool_cfg = dict(
        data_root='data/ForestDamages/active_learning_cascade_random',
        ann_file='annotations/instances_labeled_train.json',
        data_prefix=dict(
            img='images_labeled_train'
        )
    ),
    inference_options=dict(
        save_results=True,
        score_thr=0.08 ,
        uncertainty_methods=['ssc'],
        selected_metric="ssc_score",
        sample_size=200,
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