from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.evaluator import Evaluator

@HOOKS.register_module()
class ActiveLearningEvalHook(Hook):
    """主动学习评估钩子，用于在评估时输出数据使用情况"""
    
    def after_val_epoch(self, runner):
        """在验证epoch结束后调用"""
        # 获取数据集统计信息
        dataset = runner.train_dataloader.dataset
        if hasattr(dataset, 'get_dataset_stats'):
            stats = dataset.get_dataset_stats()
            total_images = stats.get('total_images', 0)
            labeled_images = stats.get('labeled_images', 0)
            usage_ratio = (labeled_images / total_images) * 100 if total_images > 0 else 0
            
            # 添加数据使用情况到日志
            runner.logger.info(
                f'Active Learning Stats - '
                f'Labeled/Total: {labeled_images}/{total_images} ({usage_ratio:.2f}%)'
            )