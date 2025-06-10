import numpy as np
from typing import Dict, List
from pycocotools.coco import COCO
import os
from sklearn.neighbors import KernelDensity
from .utils import _calculate_sor, _get_box_centers
from tqdm import tqdm

class SSCCalculator:
    def __init__(self, data_cfg: Dict):
        self.data_root = data_cfg['data_root']
        self.image_dir = os.path.join(self.data_root, data_cfg['data_prefix']['img'])
        self.annotation_file = os.path.join(self.data_root, data_cfg['ann_file'])
        self.coco = COCO(self.annotation_file)
        self.REFERENCE_SIZE = 1590

        
        # 初始化时计算整个训练集的统计信息
        self.dataset_stats = self._calculate_dataset_statistics()
    def load_coco_data(self) -> List[Dict]:
        """加载COCO格式数据"""
        img_ids = self.coco.getImgIds()
        selected_images = []
        
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.image_dir, img_info['file_name'])
            
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            bboxes = []
            labels = []
            for ann in anns:
                bbox = ann['bbox']  # [x, y, width, height]
                # 转换为[x1, y1, x2, y2]格式
                bboxes.append([
                    bbox[0], bbox[1], 
                    bbox[0] + bbox[2], 
                    bbox[1] + bbox[3]
                ])
                labels.append(ann['category_id'])
            
            if bboxes:
                selected_images.append({
                    'image_id': img_id,
                    'file_name': img_info['file_name'],
                    'bboxes': np.array(bboxes),
                    'labels': np.array(labels)
                })
        
        return selected_images
    
    def _calculate_dataset_statistics(self) -> Dict:
        """计算整个标注训练集的统计信息"""
        images = self.load_coco_data()
        detection_counts = [len(img_data['bboxes']) for img_data in images]
        
        return {
            'mean': float(np.mean(detection_counts)) if detection_counts else 0.0,
            'std': float(np.std(detection_counts)) if detection_counts else 0.0,
            'min': int(np.min(detection_counts)) if detection_counts else 0,
            'max': int(np.max(detection_counts)) if detection_counts else 0
        }

    def calculate_image_ssc(self, image_data: Dict) -> Dict[str, float]:
        bboxes = image_data['bboxes']
        labels = image_data['labels']
        
        if len(bboxes) == 0:
            return {
                'ssc_score': 0.0,
                'occlusion_score': 0.0,
                'crown_count_score': 0.0,
                'diversity_score': 0.0,
                'area_var_score': 0.0,
                'density_var_score': 0.0
            }
        # 1. 遮挡系数计算
        or_values = [_calculate_sor(bboxes[i], bboxes) for i in range(len(bboxes))]
        nb_i = len(bboxes)
        if nb_i <= 5:
            k = nb_i
        else:
            alpha = 0.3  # 对应文档中的 w_OR
            k = int(alpha * nb_i)
        k = max(1, k)  # 确保 k 至少为 1

        sorted_or = sorted(or_values, reverse=True)
        occlusion_score = np.mean(sorted_or[:k]) if k > 0 else 0.0


        # 2. 树冠数量控制系数计算 - 使用实际统计信息
        crown_count = len(bboxes)
        
        # 使用实际的数据集统计
        n_batch = self.dataset_stats['mean']  # 使用实际平均值
        sigma_batch = self.dataset_stats['std']  # 使用实际标准差
        
        # 设置边界阈值
        upper_threshold = self.dataset_stats['max'] * 2.0
        lower_threshold = self.dataset_stats['min'] * 0.3
        
        # 计算基础高斯得分
        normalized_diff = (crown_count - n_batch) / (sigma_batch + 1e-6)  # 防止除零
        base_score = np.exp(-0.5 * normalized_diff ** 2)
        
        # 平滑过渡
        transition_width = 25
        slope = 8.0
        
        # 上下界平滑过渡
        upper_smooth = 1 / (1 + np.exp((crown_count - upper_threshold + transition_width/2)/transition_width*slope))
        lower_smooth = 1 / (1 + np.exp((-crown_count + lower_threshold + transition_width/2)/transition_width*slope))
        
        # 综合得分
        crown_count_score = base_score * upper_smooth * lower_smooth
        
        # 确保分数不低于最小值
        # min_score = 0.00
        # crown_count_score = max(min_score, crown_count_score)

        # 3. 类别多样性系数计算
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        label_proportions = label_counts / len(labels)
        category_factor = np.log(1 + len(unique_labels))
        entropy = np.sum(label_proportions * np.log(label_proportions + 1e-6))
        diversity_score = category_factor * entropy
        
        # 4. 边界框面积变异系数计算
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        area_var = np.var(areas) if len(areas) > 0 else 0.0
        
        max_area = self.REFERENCE_SIZE * self.REFERENCE_SIZE
        n = len(areas)
        if n > 0:
            n_half = n // 2
            hypothetical_areas = np.concatenate([
                np.full(n_half, max_area),
                np.zeros(n - n_half)
            ])
            sigma_a0 = np.var(hypothetical_areas)
        else:
            sigma_a0 = 1.0
        
        area_var_score = (area_var / (sigma_a0 + 1e-6))

        # 5. 局部空间密度变异系数计算
        centers = _get_box_centers(bboxes)
        if len(centers) > 1:
            centers_normalized = centers / self.REFERENCE_SIZE
            kde = KernelDensity(kernel='epanechnikov', bandwidth=0.1).fit(centers_normalized)
            log_densities = kde.score_samples(centers_normalized)
            densities = np.exp(log_densities)
            normalized_densities = (densities - np.min(densities)) / (np.max(densities) - np.min(densities) + 1e-6)
            density_mean = np.mean(normalized_densities) + 1e-6
            density_var_score = np.std(normalized_densities) / density_mean
        else:
            density_var_score = 0.0

        # 综合评分计算
        # weights = [1, 0.5, 2, 0.5, 1] 
        # weights = [2, 1, 1, 1, 1] 
        weights = [1, 1, 1, 1, 2]  
        scores = [
            occlusion_score,
            crown_count_score,
            diversity_score,
            area_var_score,
            density_var_score
        ]
        
        ssc_score = float(np.dot(weights, scores))
        
        # scale_factors = {
        #     'ssc_score': 1,
        #     'occlusion_score': 1.0,
        #     'crown_count_score': 100,
        #     'diversity_score': 100,
        #     'area_var_score': 1000,
        #     'density_var_score': 10
        # }
        scale_factors = {
            'ssc_score': 1,
            'occlusion_score': 1,
            'crown_count_score': 1,
            'diversity_score': 1,
            'area_var_score': 1,
            'density_var_score': 1
        }
        # # 取小数点后4位
        # return {
        #     'ssc_score': round(ssc_score * scale_factors['ssc_score'], 2),
        #     'occlusion_score': round(float(occlusion_score) * scale_factors['occlusion_score'], 2),
        #     'crown_count_score': round(float(crown_count_score) * scale_factors['crown_count_score'], 2),
        #     'diversity_score': round(float(diversity_score) * scale_factors['diversity_score'], 2),
        #     'area_var_score': round(float(area_var_score) * scale_factors['area_var_score'], 2),
        #     'density_var_score': round(float(density_var_score) * scale_factors['density_var_score'], 2)
        # }
        # 在SSC.py中修改返回部分
        return {
            'ssc_score': int(ssc_score * scale_factors['ssc_score'] * 100) / 100,
            'occlusion_score': int(float(occlusion_score) * scale_factors['occlusion_score'] * 100) / 100,
            'crown_count_score': int(float(crown_count_score) * scale_factors['crown_count_score'] * 100) / 100,
            'diversity_score': int(float(-diversity_score) * scale_factors['diversity_score'] * 100) / 100,  # 保留负号
            'area_var_score': int(float(area_var_score) * scale_factors['area_var_score'] * 100) / 100,
            'density_var_score': int(float(density_var_score) * scale_factors['density_var_score'] * 100) / 100
        }

    

    def calculate_dataset_stats(self) -> Dict:
        """计算整个数据集的SSC统计信息"""
        images = self.load_coco_data()
        
        # 收集所有特征，包括ssc_score
        features = {
            'ssc_score': [],
            'occlusion_score': [],
            'crown_count_score': [],
            'diversity_score': [],
            'area_var_score': [],
            'density_var_score': []
        }
        
        # 添加进度条
        pbar = tqdm(total=len(images), desc="计算SSC分数")
        
        # 计算每张图片的SSC分数
        for image_data in images:
            try:
                scores = self.calculate_image_ssc(image_data)
                for name in features.keys():
                    features[name].append(scores[name])
            except Exception as e:
                self.logger.error(f"处理图片 {image_data['file_name']} 时出错: {e}")
            
            pbar.update(1)
        
        pbar.close()
        
        return {
            'features': {
                name: {
                    'mean': float(np.mean(values)) if values else 0.0,
                    'std': float(np.std(values)) if values else 0.0,
                    'min': float(np.min(values)) if values else 0.0,
                    'max': float(np.max(values)) if values else 0.0,
                    'distribution': values
                }
                for name, values in features.items()
            },
            'batch_statistics': self.dataset_stats
        }