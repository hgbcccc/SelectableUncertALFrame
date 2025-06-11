active_learning = dict(
    data_root='data/ForestDamages/active_learning_cascade_sor',
    ann_file='data/ForestDamages/active_learning_cascade_sor/annotations/instances_unlabeled.json',
    data_prefix=dict(img='data/ForestDamages/active_learning_cascade_sor/images_unlabeled'),
    # 训练池的SSC计算配置   
    train_pool_cfg = dict(
        data_root='data/ForestDamages/active_learning_cascade_sor',
        ann_file='annotations/instances_labeled_train.json',
        data_prefix=dict(
            img='images_labeled_train'
        )
    ),
    inference_options=dict(
        save_results=True,
        score_thr=0.08,
        uncertainty_methods=['sor'],  # 主方法
        selected_metric = 'sum_sor',
        sample_size=0,
        batch_size=2
    ),
    max_iterations=16,
    sample_selection=dict(
        num_samples=200,
        uncertainty_metric='sum_sor',  # 子方法
        sample_selector="default", 
    )
)
