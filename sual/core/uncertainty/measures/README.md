随机采样并不依赖于teacher模型推理结果的，置信度scores，bbox框，类别class，实现random采样的代码位于sual\core\datasets\activate_datasets.py
            if uncertainty_metric == 'random':
                if not unlabeled_images:
                    self.logger.error("没有未标注的样本可供选择")
                    return []
                    
                # 随机选择指定数量的样本
                num_samples = min(num_samples, len(unlabeled_images))
                selected = np.random.choice(
                    list(unlabeled_images), 
                    size=num_samples, 
                    replace=False
                ).tolist()
                
                # 打印选择的样本信息
                self.logger.info(f"随机选择了 {len(selected)} 个样本:")
                for img_name in selected:
                    self.logger.info(f"- {img_name}")
                    
                return selected