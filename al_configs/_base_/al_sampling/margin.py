active_learning = dict(
    data_root='data/ForestDamages/active_learning_cascade_margin',
    ann_file='data/ForestDamages/active_learning_cascade_margin/annotations/instances_unlabeled.json',
    data_prefix=dict(img='data/ForestDamages/active_learning_cascade_margin/images_unlabeled'),
    # 训练池的SSC计算配置   
    train_pool_cfg = dict(
        data_root='data/ForestDamages/active_learning_cascade_margin',
        ann_file='annotations/instances_labeled_train.json',
        data_prefix=dict(
            img='images_labeled_train'
        )
    ),
    inference_options=dict(
        save_results=True,
        score_thr=0.08 ,
        uncertainty_methods=['margin'],  # 主方法
        selected_metric = 'mean_margin',
        sample_size=0,
        batch_size=16
    ),
    max_iterations=16,
    sample_selection=dict(
        num_samples=200,
        uncertainty_metric='mean_margin',  # 子方法
        sample_selector="default", 
    )
)
