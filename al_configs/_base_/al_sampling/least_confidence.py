# score_threshold = 0.08
# max_boxes_per_img = 250
# nms_iou_threshold = 0.4

active_learning = dict(
    data_root='data/ForestDamages/active_learning_cascade_least_confidence',
    ann_file='data/ForestDamages/active_learning_cascade_least_confidence/annotations/instances_unlabeled.json',
    data_prefix=dict(img='data/ForestDamages/active_learning_cascade_least_confidence/images_unlabeled'),
    # 训练池的SSC计算配置   
    train_pool_cfg = dict(
        data_root='data/ForestDamages/active_learning_cascade_least_confidence',
        ann_file='annotations/instances_labeled_train.json',
        data_prefix=dict(
            img='images_labeled_train'
        )
    ),
    inference_options=dict(
        save_results=True,
        score_thr=0.08 ,
        uncertainty_methods=['least_confidence'],  # 主方法
        selected_metric = 'mean_least_confidence',
        sample_size=0,
        batch_size=16
    ),
    max_iterations=16,
    sample_selection=dict(
        num_samples=200,
        uncertainty_metric='mean_least_confidence',  # 子方法
        sample_selector="default", 
    ))
