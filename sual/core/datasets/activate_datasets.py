import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path  
from datetime import datetime
from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS






@DATASETS.register_module()
class ActiveCocoDataset(CocoDataset):
    def __init__(
            self,
            data_root: str,
            ann_file: str,
            data_prefix: dict,
            filter_cfg: Optional[dict] = None,
            pipeline: Optional[List[dict]] = None,
            backend_args: Optional[dict] = None,
            init_mode: bool = False,  # 添加初始化模式参数
            **kwargs
    ):
        self.init_mode = init_mode
        
        # 设置基础路径
        self.data_root = Path(data_root)
        
        # 设置目录结构
        self.ann_dir = self.data_root / 'annotations'
        self.img_dir = self.data_root
        
        # 设置图片子目录
        self.img_labeled_train = self.img_dir / 'images_labeled_train'
        self.img_labeled_val = self.img_dir / 'images_labeled_val'
        self.img_unlabeled = self.img_dir / 'images_unlabeled'
        
        # 设置标注文件路径
        self.labeled_ann_file = self.ann_dir / 'instances_labeled_train.json'
        self.unlabeled_ann_file = self.ann_dir / 'instances_unlabeled.json'
        self.val_ann_file = self.ann_dir / 'instances_labeled_val.json'
        
        # 确保目录存在
        self.ann_dir.mkdir(parents=True, exist_ok=True)
        self.img_labeled_train.mkdir(parents=True, exist_ok=True)
        self.img_labeled_val.mkdir(parents=True, exist_ok=True)
        self.img_unlabeled.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志记录器
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if self.init_mode:
            # 如果是初始化模式，创建空的标注文件
            self._create_empty_annotations()
            
        # 调用父类初始化
        super().__init__(
            ann_file=ann_file,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            pipeline=pipeline,
            backend_args=backend_args,
            **kwargs
        )
        
        # 加载标注信息
        self._load_annotations()
        
    def _create_empty_annotations(self):
        """创建空的标注文件"""
        empty_ann = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 1, 'name': 'eucalyptus', 'supercategory': 'plant'}
            ]
        }
            
        # 创建空的已标注和未标注文件
        if not self.labeled_ann_file.exists():
            with open(self.labeled_ann_file, 'w', encoding='utf-8') as f:
                json.dump(empty_ann, f, indent=2, ensure_ascii=False)
                
        if not self.unlabeled_ann_file.exists():
            with open(self.unlabeled_ann_file, 'w', encoding='utf-8') as f:
                json.dump(empty_ann, f, indent=2, ensure_ascii=False)
        
    def _load_annotations(self):
        """加载现有的标注信息"""
        try:
            with open(self.labeled_ann_file, 'r', encoding='utf-8') as f:
                self.labeled_ann = json.load(f)
            with open(self.unlabeled_ann_file, 'r', encoding='utf-8') as f:
                self.unlabeled_ann = json.load(f)
            self.logger.info("成功加载标注文件")
        except Exception as e:
            self.logger.error(f"加载标注文件失败: {str(e)}")
            raise


    def select_samples(
        self,
        results: Optional[Dict[str, Dict]] = None, 
        num_samples: int = 10,
        uncertainty_metric: str = 'entropy',
        **kwargs
    ) -> List[str]:
        """根据不确定性度量结果或随机方式选择样本"""
        # 这个选择的方法已经在sual/core/balancedscorer/defaultSelector.py中实现，且在sual/active_learnning_train_loop.py中调用
        try:
            # 打印输入信息
            self.logger.info(f"开始选择样本:")
            self.logger.info(f"- 在未标注样本中使用的不确定性排序的值为: {uncertainty_metric}")
            self.logger.info(f"- 选择的样本数: {num_samples}")

            # 检查未标注数据集中的图片
            unlabeled_images = {img['file_name'] for img in self.unlabeled_ann['images']}
            
            # 如果有 results，则使用其长度
            if results is not None:
                self.logger.info(f"- 总推理结果数: {len(results)}")
            else:
                self.logger.info(f"- 总推理结果数: 0 (未提供结果)")

            self.logger.info(f"未标注数据集中的图片数量: {len(unlabeled_images)}")

            ############################## Core-set 采样############################################
            if uncertainty_metric == 'coreset':
                self.logger.info("Core-set采样策略")
                # 收集候选样本
                valid_images = [img_name for img_name in results.keys() 
                                if img_name in unlabeled_images] if results else list(unlabeled_images)
                
                if not valid_images:
                    self.logger.warning("Core-set候选样本为空")
                    return []
                
                if not self.feature_extractor:
                    raise RuntimeError("Core-set特征提取器未初始化，请检查配置")
                
                self.logger.info("执行Core-set选择:")
                self.logger.debug(f"候选样本数: {len(valid_images)}")
                
                # 执行选择
                selector = CoreSetSelector(self.feature_extractor)
                selected = selector.select(
                    dataset=self,
                    candidates=valid_images,
                    budget=num_samples  # 直接使用配置中的num_samples
                )
                
                # 记录结果
                self.logger.info(f"完成Core-set选择，选中{len(selected)}个样本")
                return selected

            ############################## 随机采样############################################
            elif uncertainty_metric == 'random':
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

            ############################### 基本的不确定采样 ######################################
            # 基于不确定性选择
            uncertainty_scores = []
            valid_images = []
            
            if results is not None:
                for img_name, info in results.items():
                    # 检查图片是否在未标注集中
                    if img_name not in unlabeled_images:
                        self.logger.warning(f"图片 {img_name} 不在未标注数据集中")
                        continue
                    
                    # 确保推理结果中存在不确定性度量的结果！！
                    if 'uncertainty' not in info:
                        self.logger.warning(f"图片 {img_name} 缺少不确定性信息")
                        continue
                    
                    uncertainty = info['uncertainty']
                    if uncertainty_metric not in uncertainty:
                        self.logger.warning(
                            f"图片 {img_name} 缺少 {uncertainty_metric} 不确定性度量")
                        continue
                    
                    score = uncertainty[uncertainty_metric]
                    if isinstance(score, dict):
                        score = np.mean(list(score.values()))
                        self.logger.debug(f"图片 {img_name} 的平均不确定性分数: {score}")
                    
                    uncertainty_scores.append(score)
                    valid_images.append(img_name)

            self.logger.info(f"找到的有效样本数量: {len(valid_images)}")
            
            if not valid_images:
                self.logger.error("没有找到有效的样本")
                return []

            # 选择不确定性最高的样本
            uncertainty_scores = np.array(uncertainty_scores)
            num_samples = min(num_samples, len(valid_images))
            selected_indices = np.argsort(uncertainty_scores)[-num_samples:]
            
            selected = [valid_images[i] for i in selected_indices]
            
            # 打印选择的样本信息
            self.logger.info(f"选择了 {len(selected)} 个样本:")
            for i, img_name in enumerate(selected):
                score = uncertainty_scores[selected_indices[i]]
                self.logger.info(f"- {img_name}: {score:.4f}")
            
            return selected
                
        except Exception as e:
            self.logger.error(f"样本选择失败: {str(e)}")
            return []

                    


        #     ############################### 基本的不确定采样 ######################################
        #     # 基于不确定性选择
        #     uncertainty_scores = []
        #     valid_images = []
        #     score_details = {}  # 添加详细分数记录
            
        #     if results is not None:
        #         for img_name, info in results.items():
        #             if img_name not in unlabeled_images:
        #                 self.logger.warning(f"图片 {img_name} 不在未标注数据集中")
        #                 continue
                    
        #             if 'uncertainty' not in info:
        #                 self.logger.warning(f"图片 {img_name} 缺少不确定性信息")
        #                 continue
                    
        #             uncertainty = info['uncertainty']
        #             if uncertainty_metric not in uncertainty:
        #                 self.logger.warning(
        #                     f"图片 {img_name} 缺少 {uncertainty_metric} 不确定性度量")
        #                 continue
                    
        #             score = uncertainty[uncertainty_metric]
        #             if isinstance(score, dict):
        #                 score = np.mean(list(score.values()))
                    
        #             # 记录完整的不确定性信息
        #             score_details[img_name] = {
        #                 'score': float(score),  # 确保可JSON序列化
        #                 'rank': None,  # 稍后填充
        #                 'selected': False,  # 是否被选中
        #                 'all_uncertainty_metrics': {
        #                     k: float(v) if isinstance(v, (int, float, np.number)) 
        #                     else v for k, v in uncertainty.items()
        #                 },  # 所有不确定性指标
        #                 'predictions': {
        #                     'num_predictions': len(info.get('predictions', [])),
        #                     'prediction_scores': [
        #                         float(p['score']) for p in info.get('predictions', [])
        #                     ] if 'predictions' in info else [],
        #                     'prediction_labels': [
        #                         p['label'] for p in info.get('predictions', [])
        #                     ] if 'predictions' in info else []
        #                 }
        #             }
                    
        #             uncertainty_scores.append(score)
        #             valid_images.append(img_name)

        #         self.logger.info(f"找到的有效样本数量: {len(valid_images)}")
                
        #         if not valid_images:
        #             self.logger.error("没有找到有效的样本")
        #             return []

        #         # 选择不确定性最高的样本
        #         uncertainty_scores = np.array(uncertainty_scores)
        #         num_samples = min(num_samples, len(valid_images))
                
        #         # 获取完整的排序索引
        #         full_ranking = np.argsort(uncertainty_scores)[::-1]  # 降序排序
                
        #         # 更新每个样本的排名信息
        #         ranked_results = []  # 存储按排名排序的结果
        #         for rank, idx in enumerate(full_ranking, 1):
        #             img_name = valid_images[idx]
        #             score_details[img_name]['rank'] = rank
        #             score_details[img_name]['selected'] = rank <= num_samples
                    
        #             # 添加到排序结果列表
        #             ranked_results.append({
        #                 'rank': rank,
        #                 'image_name': img_name,
        #                 'score': float(uncertainty_scores[idx]),
        #                 'selected': rank <= num_samples,
        #                 **score_details[img_name]  # 包含所有详细信息
        #             })
                
        #         # 选择top-k样本
        #         selected_indices = full_ranking[:num_samples]
        #         selected = [valid_images[i] for i in selected_indices]
                
        #         # 保存排序结果到文件
        #         ranking_info = {
        #             'metadata': {
        #                 'metric': uncertainty_metric,
        #                 'total_samples': len(valid_images),
        #                 'selected_samples': num_samples,
        #                 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #             },
        #             'statistics': {
        #                 'mean_score': float(np.mean(uncertainty_scores)),
        #                 'std_score': float(np.std(uncertainty_scores)),
        #                 'min_score': float(np.min(uncertainty_scores)),
        #                 'max_score': float(np.max(uncertainty_scores)),
        #                 'median_score': float(np.median(uncertainty_scores)),
        #                 'quartiles': {
        #                     'q25': float(np.percentile(uncertainty_scores, 25)),
        #                     'q75': float(np.percentile(uncertainty_scores, 75))
        #                 },
        #                 'selected_stats': {
        #                     'mean_score': float(np.mean(uncertainty_scores[selected_indices])),
        #                     'std_score': float(np.std(uncertainty_scores[selected_indices]))
        #                 }
        #             },
        #             'detailed_results': {
        #                 'all_samples': score_details,  # 所有样本的详细信息
        #                 'ranked_list': ranked_results,  # 按排名排序的结果
        #                 'selected_samples': selected  # 最终选择的样本
        #             }
        #         }
                
        #         # 创建排序结果目录
        #         ranking_dir = self.data_root / 'ranking_results'
        #         ranking_dir.mkdir(exist_ok=True)
                
        #         # 保存排序结果
        #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #         ranking_file = ranking_dir / f'ranking_{uncertainty_metric}_{timestamp}.json'
        #         with open(ranking_file, 'w', encoding='utf-8') as f:
        #             json.dump(ranking_info, f, indent=2, ensure_ascii=False)
                
        #         # 打印选择的样本信息
        #         self.logger.info(f"\n选择了 {len(selected)} 个样本:")
        #         self.logger.info("\n排序详细信息:")
        #         self.logger.info(f"{'排名':<6}{'图片名称':<50}{'分数':<10}{'是否选中':<10}")
        #         self.logger.info("-" * 76)
                
        #         # 打印前20个样本的信息（包括选中和未选中的）
        #         for result in ranked_results[:20]:
        #             self.logger.info(
        #                 f"{result['rank']:<6}{result['image_name']:<50}"
        #                 f"{result['score']:.4f}  {'√' if result['selected'] else '×'}"
        #             )
                
        #         return selected
                
        # except Exception as e:
        #     self.logger.error(f"样本选择失败: {str(e)}")
        #     return []


    def update_dataset(
            self,
            selected_samples: List[str],
            new_annotations: Optional[Dict] = None
    ) -> bool:
        """更新数据集标注信息"""
        try:
            if not selected_samples:
                self.logger.warning("没有选中的样本需要更新")
                return False
            
            # 如果没有提供新标注，从unlabeled_ann中提取
            if new_annotations is None:
                new_annotations = {
                    'images': [],
                    'annotations': []
                }
                
                # 获取选中样本的图片ID
                selected_img_ids = set()
                for img in self.unlabeled_ann['images']:
                    if img['file_name'] in selected_samples:
                        selected_img_ids.add(img['id'])
                        new_annotations['images'].append(img)
                
                # 获取对应的标注
                for ann in self.unlabeled_ann['annotations']:
                    if ann['image_id'] in selected_img_ids:
                        new_annotations['annotations'].append(ann)
            
            # 更新已标注数据集
            self.labeled_ann['images'].extend(new_annotations['images'])
            self.labeled_ann['annotations'].extend(new_annotations['annotations'])
            
            # 更新未标注数据集
            remaining_images = []
            remaining_anns = []
            selected_img_ids = {img['id'] for img in new_annotations['images']}
            
            for img in self.unlabeled_ann['images']:
                if img['id'] not in selected_img_ids:
                    remaining_images.append(img)
                    
            for ann in self.unlabeled_ann['annotations']:
                if ann['image_id'] not in selected_img_ids:
                    remaining_anns.append(ann)
                    
            self.unlabeled_ann['images'] = remaining_images
            self.unlabeled_ann['annotations'] = remaining_anns
            
            # 保存更新后的标注文件
            with open(self.labeled_ann_file, 'w', encoding='utf-8') as f:
                json.dump(self.labeled_ann, f, indent=2, ensure_ascii=False)
            
            with open(self.unlabeled_ann_file, 'w', encoding='utf-8') as f:
                json.dump(self.unlabeled_ann, f, indent=2, ensure_ascii=False)
            
            # 移动图片文件
            for img_name in selected_samples:
                src_path = self.img_unlabeled / img_name
                dst_path = self.img_labeled_train / img_name
                if src_path.exists():
                    shutil.move(str(src_path), str(dst_path))
                else:
                    self.logger.warning(f"找不到源文件: {src_path}")
            
            self.logger.info(f"成功更新数据集，移动了 {len(selected_samples)} 个样本")
            return True
            
        except Exception as e:
            self.logger.error(f"更新数据集失败: {str(e)}")
            return False

    def get_dataset_stats(self) -> Dict:
        """获取数据集统计信息"""
        try:
            stats = {
                'total_images': len(self.labeled_ann['images']) + len(self.unlabeled_ann['images']),
                'labeled_images': len(self.labeled_ann['images']),
                'unlabeled_images': len(self.unlabeled_ann['images']),
                'labeled_annotations': len(self.labeled_ann['annotations']),
                'unlabeled_annotations': len(self.unlabeled_ann['annotations']),
                'labeled_ratio': len(self.labeled_ann['images']) / (
                    len(self.labeled_ann['images']) + len(self.unlabeled_ann['images'])
                )
            }
            
            # 添加每个类别的统计信息
            category_stats = {}
            for cat in self.labeled_ann['categories']:
                cat_id = cat['id']
                cat_name = cat['name']
                labeled_count = len([
                    ann for ann in self.labeled_ann['annotations']
                    if ann['category_id'] == cat_id
                ])
                unlabeled_count = len([
                    ann for ann in self.unlabeled_ann['annotations']
                    if ann['category_id'] == cat_id
                ])
                category_stats[cat_name] = {
                    'labeled': labeled_count,
                    'unlabeled': unlabeled_count,
                    'total': labeled_count + unlabeled_count
                }
            
            stats['category_stats'] = category_stats
            return stats
            
        except Exception as e:
            self.logger.error(f"获取数据集统计信息失败: {str(e)}")
            return {}

# #############################如下的模块因为循环导入问题所以在此##################


class FeatureExtractor:
    """特征提取器模块"""
    def __init__(self, backbone: nn.Module, img_size: tuple = (224, 224), cache_dir: str = None):
        self.backbone = backbone
        self.img_size = img_size
        self.preprocess = self._build_preprocess(),
        self.normalize = nn.Sequential(
            nn.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet标准化
                        std=[0.229, 0.224, 0.225])
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None,
        self._load_cache()
        
    def _build_preprocess(self) -> nn.Module:
        """构建预处理管道"""
        return nn.Sequential(nn.Unflatten(0, (1, 3, self.img_size[0], self.img_size[1]))) # 假设输入为CHW格式 # 根据实际需要添加归一化层等
    
    def extract(self, img_path: str) -> torch.Tensor:
        """从图像路径提取特征"""
        # 加载图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        
        # 转换为tensor并预处理
        tensor = self.normalize(tensor)
        tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # HWC -> CHW
        tensor = self.preprocess(tensor.unsqueeze(0))  # 增加batch维度
        
        # 特征提取
        with torch.no_grad():
            features = self.backbone(tensor)
        return features.flatten()

class FeatureExtractorFactory:
    """动态创建特征提取器"""
    @classmethod
    def from_config(cls, cfg: dict) -> FeatureExtractor:
        import torchvision
        
        # 加载预训练模型
        model = getattr(torchvision.models, cfg['backbone'])(
            weights='DEFAULT' if cfg['pretrained'] else None
        )
        
        # 构建特征提取层
        feature_extractor = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
            model.avgpool,
            nn.Flatten()  # 新增展平层
        )
        
        return FeatureExtractor(
            backbone=feature_extractor,
            img_size=cfg['img_size'],
            # cache_dir=cfg['cache_dir'],# 有一个错误
            cache_dir=cfg.get('cache_dir')#

        )



class CoreSetSelector:
    """Core-set样本选择模块"""
    def __init__(self, feature_extractor: FeatureExtractor):
        #正确的导入位置，在使用时候进行导入

        
        self.feature_extractor = feature_extractor
        self.feature_cache = {}
    
    def _get_image_path(self, dataset: ActiveCocoDataset, img_name: str) -> str:

        """获取完整图像路径"""
        return str(dataset.img_unlabeled / img_name)
    
    def _compute_features(self, dataset: ActiveCocoDataset, img_names: List[str]) -> np.ndarray:
        """批量计算特征"""
        features = []
        for name in img_names:
            if name not in self.feature_cache:
                img_path = self._get_image_path(dataset, name)
                self.feature_cache[name] = self.feature_extractor.extract(img_path)
            features.append(self.feature_cache[name].cpu().numpy())
        return np.array(features)
    
    def select(self, dataset: ActiveCocoDataset, candidates: List[str], budget: int) -> List[str]:
        """执行Core-set选择"""
        if len(candidates) <= budget:
            return candidates.copy()
        
        # 计算特征
        features = self._compute_features(dataset, candidates)
        features = np.nan_to_num(features)   # 可能有问题
        features = np.clip(features, -1e5, 1e5)  # 防止溢出   # 可能有问题
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        # 贪婪k-center算法
        selected = []
        remaining_indices = list(range(len(candidates)))
        
        # 初始化第一个样本
        first_idx = np.random.choice(len(candidates))
        selected.append(first_idx)
        remaining_indices.remove(first_idx)
        
        for _ in range(budget-1):
            # 计算最小距离
            dists = np.linalg.norm(
                features[remaining_indices] - features[selected][:, None], 
                axis=2
            )
            min_dists = np.min(dists, axis=0)
            
            # 选择最远点
            farthest = np.argmax(min_dists)
            selected_idx = remaining_indices[farthest]
            
            selected.append(selected_idx)
            remaining_indices.pop(farthest)
        
        return [candidates[i] for i in selected]