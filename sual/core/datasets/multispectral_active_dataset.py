from typing import List, Dict, Optional, Tuple
import numpy as np
from mmdet.datasets import CocoDataset
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import rasterio
from scipy.spatial.distance import cdist
import json
import logging
import shutil
from pathlib import Path
from sual.core.uncertainty.spectral_metrics import SpectralFeatureExtractor, SpectralSamplingStrategy




class MultispectralActiveDataset(CocoDataset):
    """多光谱主动学习数据集类"""
    
    def __init__(
            self,
            data_root: str,
            ann_file: str,
            data_prefix: dict,
            filter_cfg: Optional[dict] = None,
            pipeline: Optional[List[dict]] = None,
            backend_args: Optional[dict] = None,
            spectral_indices: List[str] = ['ndvi', 'ndwi', 'evi'],
            sampling_strategy: str = 'cluster',
            distance_metric: str = 'euclidean',
            init_mode: bool = False,
            **kwargs
    ):
        """初始化多光谱主动学习数据集
        
        Args:
            data_root: 数据根目录
            ann_file: 标注文件路径
            data_prefix: 数据前缀配置
            filter_cfg: 过滤配置
            pipeline: 数据处理流水线
            backend_args: 后端参数
            spectral_indices: 光谱指数列表
            sampling_strategy: 采样策略
            distance_metric: 距离度量方法
            init_mode: 是否为初始化模式
            **kwargs: 其他参数
        """
        self.init_mode = init_mode
        self.spectral_indices = spectral_indices
        self.sampling_strategy = sampling_strategy
        self.distance_metric = distance_metric
        
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
        
        # 初始化光谱特征提取器
        self.feature_extractor = SpectralFeatureExtractor(
            selected_indices=spectral_indices,
            index_weights={index: 1.0 for index in spectral_indices}
        )
        
        # 初始化采样策略
        self.sampler = SpectralSamplingStrategy()
        
        if self.init_mode:
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

    def _extract_features(self, img: np.ndarray) -> np.ndarray:
        """提取单个图像的特征
        
        Args:
            img: 输入图像 (H, W, C)
            
        Returns:
            np.ndarray: 特征向量
        """
        try:
            features = self.feature_extractor.extract_features(
                images=img[np.newaxis, ...],
                normalize=True
            )
            return features[0]
        except Exception as e:
            self.logger.error(f"特征提取失败: {str(e)}")
            raise

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

    
    def select_samples(self, 
                    num_samples: int = 10,
                    **sampling_kwargs) -> List[str]:
        """基于光谱特征选择样本
        
        Args:
            num_samples: 需要选择的样本数
            **sampling_kwargs: 采样策略的其他参数
            
        Returns:
            List[str]: 选中的样本文件名列表
        """
        try:
            self.logger.info(f"开始选择样本:")
            self.logger.info(f"- 策略: {self.sampling_strategy}")
            self.logger.info(f"- 距离度量: {self.distance_metric}")
            self.logger.info(f"- 需要选择的样本数: {num_samples}")
            
            # 获取未标注图像列表
            unlabeled_images = {img['file_name'] for img in self.unlabeled_ann['images']}
            if not unlabeled_images:
                self.logger.warning("未标注数据集为空")
                return []
                
            self.logger.info(f"未标注数据集中的图片数量: {len(unlabeled_images)}")
            
            # 收集所有未标注图像的特征
            features_dict = {}
            for img_info in self.unlabeled_ann['images']:
                img_path = self.img_unlabeled / img_info['file_name']
                try:
                    with rasterio.open(str(img_path)) as src:
                        img = src.read().transpose(1, 2, 0)
                    features = self._extract_features(img)
                    features_dict[img_info['file_name']] = features
                except Exception as e:
                    self.logger.warning(f"处理图像 {img_info['file_name']} 失败: {str(e)}")
                    continue
            
            if not features_dict:
                self.logger.error("没有成功提取任何特征")
                return []
            
            # 转换为特征矩阵
            image_names = list(features_dict.keys())
            features = np.stack([features_dict[name] for name in image_names])
            
            # 使用采样策略选择样本
            selected_indices = self.sampler.sample(
                features=features,
                n_samples=min(num_samples, len(features)),
                strategy=self.sampling_strategy,
                distance_metric=self.distance_metric,
                **sampling_kwargs
            )
            
            # 获取选中的文件名
            selected_samples = [image_names[i] for i in selected_indices]
            
            # 计算采样评估指标
            metrics = self.sampler.evaluate_sampling(
                features=features,
                selected_indices=selected_indices,
                distance_metric=self.distance_metric
            )
            
            self.logger.info("采样评估指标:")
            for metric_name, value in metrics.items():
                self.logger.info(f"- {metric_name}: {value:.4f}")
            
            self.logger.info(f"选择了 {len(selected_samples)} 个样本")
            return selected_samples
            
        except Exception as e:
            self.logger.error(f"样本选择失败: {str(e)}")
            raise

    def update_feature_extractor(self,
                               index_weights: Optional[Dict[str, float]] = None,
                               selected_indices: Optional[List[str]] = None):
        """更新特征提取器配置"""
        try:
            if index_weights is not None:
                self.feature_extractor.set_index_weights(index_weights)
            if selected_indices is not None:
                self.feature_extractor.set_selected_indices(selected_indices)
            self.logger.info("特征提取器配置已更新")
        except Exception as e:
            self.logger.error(f"更新特征提取器配置失败: {str(e)}")
            raise
        
    def set_sampling_strategy(self,
                            strategy: str,
                            distance_metric: Optional[str] = None,
                            **strategy_params):
        """设置采样策略和参数"""
        try:
            if strategy not in self.sampler.strategies:
                raise ValueError(f"不支持的采样策略: {strategy}")
            self.sampling_strategy = strategy
            
            if distance_metric is not None:
                if distance_metric not in self.sampler.distance_metrics:
                    raise ValueError(f"不支持的距离度量方法: {distance_metric}")
                self.distance_metric = distance_metric
            
            self.sampler.set_strategy_params(strategy, **strategy_params)
            self.logger.info(f"采样策略已更新为: {strategy}")
            if distance_metric:
                self.logger.info(f"距离度量方法已更新为: {distance_metric}")
        except Exception as e:
            self.logger.error(f"设置采样策略失败: {str(e)}")
            raise
        
    def update_dataset(self, selected_samples: List[str]) -> bool:
        """更新数据集标注信息
        
        Args:
            selected_samples: 选中的图像文件名列表
            
        Returns:
            bool: 更新是否成功
        """
        try:
            if not selected_samples:
                self.logger.warning("没有选中的样本需要更新")
                return False
            
            # 获取选中样本的图片ID
            selected_img_ids = set()
            new_annotations = {
                'images': [],
                'annotations': []
            }
            
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

