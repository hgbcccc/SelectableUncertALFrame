# MUS-CDB (Model Uncertainty Sampling + Class Diversity Balancing) 采样策略配置
# 基于模型不确定性采样和类别多样性平衡的两阶段主动学习采样

active_learning = dict(
    data_root='data/ForestDamages/active_learning_cascade_mus_cdb',
    ann_file='data/ForestDamages/active_learning_cascade_mus_cdb/annotations/instances_unlabeled.json',
    data_prefix=dict(img='data/ForestDamages/active_learning_cascade_mus_cdb/images_unlabeled'),
    # 训练池的SSC计算配置   
    train_pool_cfg = dict(
        data_root='data/ForestDamages/active_learning_cascade_mus_cdb',
        ann_file='annotations/instances_labeled_train.json',
        data_prefix=dict(
            img='images_labeled_train'
        )
    ),
    inference_options=dict(
        save_results=True,
        score_thr=0.08,
        uncertainty_methods=['mus'],  # 主方法
        # selected_metric = 'sum_mus',
        sample_size=0,
        batch_size=32  # 增加批量大小以提升性能
    ),
    max_iterations=16,
    sample_selection=dict(
        num_samples=200,
        # uncertainty_metric='sum_mus',  # 子方法
        sample_selector="default", 
        uncertainty_metric='mus',  # 关键：指定使用MUS指标排序
    ))

