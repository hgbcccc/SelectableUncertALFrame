import argparse
import os
import numpy as np
from pathlib import Path
import json
import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from sual.inference.simdetector import SimDetectionInference
from sual.core.datasets import ActiveCocoDataset
import os.path as osp
import re
from mmengine.logging import MMLogger
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sual.core.uncertainty.measures.ssc_calculator import SSCCalculator
from sual.core.balancedscorer import WassersteinSampleSelector
from sual.core.balancedscorer import RLSampleSelector
from sual.core.balancedscorer import CombinatorialSelector
"""
主动学习训练脚本

该脚本实现了一个主动学习的训练流程，主要包括以下步骤：

1. **推理训练集**：
   使用目标检测模型对标注的训练集进行推理，得到每张图片的检测结果和不确定性指标。

   示例数据结构：
   result_un = {
       'image1.jpg': {
           'result': DetDataSample对象,  # 检测结果
           'uncertainty': {
               'occlusion_score': 28.20,  # 遮挡分数
               'crown_count_score': 3.41e-11,  # 树冠数量控制分数
               'diversity_score': 0.038,  # 多样性分数
               'area_var_score': 2.0,  # 面积变异分数
               'density_var_score': 0.91,  # 密度变异分数
               'ssc_score': 10.0  # 原始空间结构复杂度分数
           }
       },
       'image2.jpg': {
           'result': DetDataSample对象,
           'uncertainty': {
               'occlusion_score': 30.50,
               'crown_count_score': 1.20e-10,
               'diversity_score': 0.045,
               'area_var_score': 1.8,
               'density_var_score': 0.85,
               'ssc_score': 12.0
           }
       }
   }

2. **计算不确定性**：
   对推理结果调用`compute_uncertainty`方法，计算每张图片的多个不确定性指标。

3. **使用WassersteinBalancedScorer**：
   创建`WassersteinBalancedScorer`实例，计算每个样本的平衡得分，并更新`ssc_score`。

   示例数据结构：
   processed_results = {
       'image1.jpg': {
           'result': DetDataSample对象,
           'vis_path': 'path/to/visualization.jpg',
           'uncertainty': {
               'occlusion_score': 28.20,
               'crown_count_score': 3.41e-11,
               'diversity_score': 0.038,
               'area_var_score': 2.0,
               'density_var_score': 0.91,
               'ssc_score': 15.0,  # 更新后的分数
               'wasserstein_balanced_score': 20.0,  # 新的平衡得分
               'w_score': 5.0,  # 线性组合得分
               'mapd_score': 0.1,  # MAPD得分
               'perturbation_score': 1.5,  # 扰动得分
               'feature_wasserstein_distances': {...}  # 各特征的Wasserstein距离
           }
       },
       'image2.jpg': {...}
   }

4. **选择样本**：
   使用`dataset.select_samples`方法，根据计算出的平衡得分选择样本。

5. **更新数据集**：
   将选中的样本更新到数据集中。

示例输出：
    result_un 中的 ssc_score 平均值: 11.0
    processed_results 中的 ssc_score 平均值: 16.5
    选择完成，选中样本数量: 5
    数据集更新成功

通过这种方式，脚本能够有效地选择未标注样本，增强模型的泛化能力。
"""

# 查找最佳检查点
def find_best_checkpoint(work_dir: Path, logger: Optional[MMLogger] = None) -> Optional[str]:
    """查找最佳检查点
    
    策略：
    1. 从日志中查找最佳检查点信息
    2. 在工作目录中查找所有检查点
    3. 按照不同类型的检查点进行优先级排序
    """
    work_dir = Path(work_dir)
    
    # 1. 从日志文件中查找最佳检查点
    log_file = work_dir / 'run.log'
    best_ckpt = None
    
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_content = f.read()
            # 使用正则表达式匹配最后一次保存的最佳检查点
            matches = re.finditer(
                r'The best checkpoint .+ is saved to (.+\.pth)',
                log_content
            )
            # 获取最后一个匹配结果
            best_ckpt_matches = list(matches)
            if best_ckpt_matches:
                best_ckpt = best_ckpt_matches[-1].group(1)
                best_ckpt = work_dir / best_ckpt
                if best_ckpt.exists():
                    if logger:
                        logger.info(f'从日志中找到最佳检查点: {best_ckpt}')
                    return str(best_ckpt)
    
    # 2. 在工作目录中查找所有检查点
    def get_checkpoint_priority(ckpt_path: Path) -> int:
        """定义检查点的优先级"""
        name = ckpt_path.name
        if 'best' in name and 'bbox_mAP' in name:
            return 4  # 最高优先级：性能最好的检查点
        if 'best' in name:
            return 3  # 其他最佳检查点
        if 'epoch' in name:
            return 2  # epoch 检查点
        return 1  # 其他检查点
    
    # 递归查找所有 .pth 文件
    checkpoints: List[Path] = []
    for ext in ['.pth', '.pt', '.ckpt']:  # 支持多种扩展名
        checkpoints.extend(work_dir.rglob(f'*{ext}'))
    
    if not checkpoints:
        if logger:
            logger.warning(f'在 {work_dir} 中未找到任何检查点')
        return None
    
    # 按优先级和修改时间排序
    checkpoints.sort(
        key=lambda x: (
            get_checkpoint_priority(x),  # 首先按优先级
            x.stat().st_mtime  # 然后按修改时间
        ),
        reverse=True
    )
    
    best_ckpt = str(checkpoints[0])
    if logger:
        logger.info(f'找到最佳检查点: {best_ckpt}')
    return best_ckpt

# 调整阈值参数
def adjust_thresholds(cfg, current_iter: int, total_iters: int):
    """调整阈值参数
    Args:
        cfg: 配置对象
        current_iter: 当前迭代次数
        total_iters: 总迭代次数
    Returns:
        tuple: (score_threshold, max_boxes_per_img, nms_iou_threshold)
    """
    progress = current_iter / total_iters
    
    # 初始值和目标值设定
    score_thr_init, score_thr_final = 0.08, 0.3
    max_per_img_init, max_per_img_final = 300, 100
    iou_threshold_init, iou_threshold_final = 0.6, 0.5
    
    # 线性调整参数
    def linear_adjust(init_val, final_val, progress):
        return init_val + (final_val - init_val) * progress
    
    # 计算新的阈值
    score_threshold = linear_adjust(score_thr_init, score_thr_final, progress)
    max_boxes_per_img = int(linear_adjust(max_per_img_init, max_per_img_final, progress))
    nms_iou_threshold = linear_adjust(iou_threshold_init, iou_threshold_final, progress)
    
    logger = MMLogger.get_current_instance()
    logger.info(
        f"调整阈值参数 - 进度: {progress:.2f}\n"
        f"score_threshold: {score_threshold:.3f}\n"
        f"max_boxes_per_img: {max_boxes_per_img}\n"
        f"nms_iou_threshold: {nms_iou_threshold:.3f}"
    )
    
    return score_threshold, max_boxes_per_img, nms_iou_threshold

# 标准化不确定性分数
def normalize_uncertainty_scores(unlabeled_pool_results):
    # Step 1: 收集所有指标的全局值
    metric_values = {}
    
    # 遍历所有图片收集指标
    for img_data in unlabeled_pool_results.values():
        if 'uncertainty' not in img_data:
            continue
        for metric, value in img_data['uncertainty'].items():
            if isinstance(value, (int, float)):
                metric_values.setdefault(metric, []).append(value)

    # Step 2: 计算各指标的min-max范围
    metric_ranges = {
        metric: {'min': min(values), 'max': max(values)}
        for metric, values in metric_values.items()
    }

    # Step 3: 创建标准化后的数据结构
    normalized_results = {}
    
    for img_name, img_data in unlabeled_pool_results.items():
        if 'uncertainty' not in img_data:
            normalized_results[img_name] = img_data
            continue

        normalized_metrics = {}
        for metric, value in img_data['uncertainty'].items():
            if metric not in metric_ranges:
                normalized_metrics[metric] = value
                continue

            v_min = metric_ranges[metric]['min']
            v_max = metric_ranges[metric]['max']
            
            # 处理除零情况
            if v_max == v_min:
                normalized = 0.5  # 当所有值相同时设为中间值
            else:
                normalized = (value - v_min) / (v_max - v_min)
                normalized = max(0.0, min(1.0, normalized))  # 确保在[0,1]范围内

            normalized_metrics[metric] = round(normalized, 4)

        # 保持原有数据结构
        normalized_results[img_name] = {
            **img_data,
            'uncertainty_normalized': normalized_metrics
        }

    return normalized_results

# 标准化训练集统计信息
def normalize_train_stats(train_stats):
    """对训练集统计信息进行全局标准化
    
    将训练集中的特征分布标准化到[0,1]范围内，便于与未标注样本进行比较
    
    Args:
        train_stats: 训练集统计信息字典
        
    Returns:
        标准化后的训练集统计信息字典
    """
    if 'features' not in train_stats:
        return train_stats
    
    normalized_train_stats = {
        'features': {},
        'batch_statistics': train_stats.get('batch_statistics', {})
    }
    
    # 对每个特征进行标准化
    for feature_name, feature_stats in train_stats['features'].items():
        if 'distribution' not in feature_stats or not feature_stats['distribution']:
            normalized_train_stats['features'][feature_name] = feature_stats
            continue
        
        # 获取分布的最小值和最大值
        distribution = feature_stats['distribution']
        v_min = min(distribution)
        v_max = max(distribution)
        
        # 处理除零情况
        if v_max == v_min:
            normalized_distribution = [0.5] * len(distribution)  # 当所有值相同时设为中间值
        else:
            # 标准化分布
            normalized_distribution = [
                max(0.0, min(1.0, (x - v_min) / (v_max - v_min)))
                for x in distribution
            ]
        
        # 计算标准化后的均值和标准差
        normalized_mean = sum(normalized_distribution) / len(normalized_distribution)
        normalized_std = np.std(normalized_distribution) if len(normalized_distribution) > 1 else 0.0
        
        # 保存标准化后的统计信息
        normalized_train_stats['features'][feature_name] = {
            'mean': normalized_mean,
            'std': normalized_std,
            'distribution': normalized_distribution,
            'min': 0.0,  # 标准化后的最小值
            'max': 1.0   # 标准化后的最大值
        }
    
    return normalized_train_stats

# 重新计算未标注池中每张图片的树冠控制系数
def recalculate_crown_count_scores(unlabeled_pool_results, unlabeled_pool_results_uncertainty):
    """重新计算未标注池中每张图片的树冠控制系数
    
    基于当前未标注池的批次统计信息，重新计算每张图片的树冠控制系数，
    并更新到不确定性结果中。
    
    Args:
        unlabeled_pool_results: 未标注池的推理结果
        unlabeled_pool_results_uncertainty: 未标注池的不确定性结果
        
    Returns:
        更新后的不确定性结果
    """
    # 计算未标注池的批次统计信息
    unlabeled_batch_stats = {
        'batch_statistics': {
            'mean': 0.0,
            'std': 0.0,
            'min': float('inf'),
            'max': float('-inf')
        }
    }
    
    # 收集所有图片的检测框数量
    crown_counts = []
    for img_name, img_data in unlabeled_pool_results.items():
        if 'result' in img_data and hasattr(img_data['result'], '_pred_instances'):
            bboxes = img_data['result']._pred_instances.bboxes
            crown_count = len(bboxes)
            crown_counts.append(crown_count)
            unlabeled_batch_stats['batch_statistics']['min'] = min(
                unlabeled_batch_stats['batch_statistics']['min'], crown_count)
            unlabeled_batch_stats['batch_statistics']['max'] = max(
                unlabeled_batch_stats['batch_statistics']['max'], crown_count)
    
    # 计算均值和标准差
    if crown_counts:
        unlabeled_batch_stats['batch_statistics']['mean'] = sum(crown_counts) / len(crown_counts)
        unlabeled_batch_stats['batch_statistics']['std'] = np.std(crown_counts) if len(crown_counts) > 1 else 0.0
    else:
        unlabeled_batch_stats['batch_statistics']['min'] = 0
        unlabeled_batch_stats['batch_statistics']['max'] = 0
    
    # 重新计算每张图片的树冠控制系数
    for img_name, img_data in unlabeled_pool_results.items():
        if 'result' in img_data and hasattr(img_data['result'], '_pred_instances'):
            # 获取检测结果
            result = img_data['result']
            bboxes = result._pred_instances.bboxes
            labels = result._pred_instances.labels
            
            # 计算树冠数量
            crown_count = len(bboxes)
            
            # 使用批次统计信息计算树冠控制系数
            stats = unlabeled_batch_stats['batch_statistics']
            n_batch = stats['mean']
            sigma_batch = stats['std']
            
            # 放宽边界阈值
            upper_threshold = stats['max'] * 2.0
            lower_threshold = stats['min'] * 0.3
            
            # 计算基础高斯得分
            normalized_diff = (crown_count - n_batch) / (sigma_batch + 1e-6)
            base_score = np.exp(-0.5 * normalized_diff ** 2)
            
            # 含过渡和平滑优化
            transition_width = 25
            slope = 8.0
            
            # 上界平滑过渡
            upper_smooth = 1 / (1 + np.exp((crown_count - upper_threshold + transition_width/2)/transition_width*slope))
            
            # 下界平滑过渡  
            lower_smooth = 1 / (1 + np.exp((-crown_count + lower_threshold + transition_width/2)/transition_width*slope))
            
            # 综合得分
            crown_count_score = base_score * upper_smooth * lower_smooth
            
            # 确保分数不低于最小值
            # min_score = 0.00
            # crown_count_score = max(min_score, crown_count_score)
            
            # 更新不确定性结果中的树冠控制系数
            if img_name in unlabeled_pool_results_uncertainty and 'uncertainty' in unlabeled_pool_results_uncertainty[img_name]:
                unlabeled_pool_results_uncertainty[img_name]['uncertainty']['crown_count_score'] = crown_count_score
    print(f"n_batch: {n_batch}, sigma_batch: {sigma_batch}")
    return unlabeled_pool_results_uncertainty

# 将未标注池数据转换为与train_stats相似的格式
def convert_unlabeled_to_stats_format(unlabeled_pool_results: Dict) -> Dict:
    """
    将未标注池数据转换为与train_stats相似的格式
    
    Args:
        unlabeled_pool_results: 未标注池数据
        {
            'img_id.jpg': {
                'uncertainty': {
                    'ssc_score': float,
                    'occlusion_score': float,
                    'crown_count_score': float,
                    'diversity_score': float,
                    'area_var_score': float,
                    'density_var_score': float
                }
            },
            ...
        }
    
    Returns:
        {
            'features': {
                'ssc_score': {
                    'mean': float,
                    'std': float,
                    'distribution': List[float],
                    'min': float,
                    'max': float
                },
                'occlusion_score': {
                    'mean': float,
                    'std': float,
                    'distribution': List[float],
                    'min': float,
                    'max': float
                },
                # 其他特征...
            },
            'image_mapping': {
                'image_names': List[str],
                'indices': Dict[str, int]
            }
        }
    """
    # 初始化特征列表 - 确保与输入数据结构匹配
    features = [
        'ssc_score',
        'occlusion_score',
        'crown_count_score',
        'diversity_score',
        'area_var_score',
        'density_var_score'
    ]
    
    # 初始化返回的数据结构
    stats = {
        'features': {
            feature: {
                'distribution': [],
                'mean': 0.0,
                'std': 0.0,
                'min': float('inf'),
                'max': float('-inf')
            } for feature in features
        },
        'image_mapping': {
            'image_names': [],
            'indices': {}
        }
    }
    
    # 收集数据
    for idx, (image_name, data) in enumerate(unlabeled_pool_results.items()):
        # 保存图片名称和索引映射
        stats['image_mapping']['image_names'].append(image_name)
        stats['image_mapping']['indices'][image_name] = idx
        
        # 收集每个特征的值 - 直接从uncertainty中获取
        uncertainty = data['uncertainty']
        for feature in features:
            # 确保特征存在于uncertainty字典中
            if feature in uncertainty:
                value = uncertainty[feature]
                stats['features'][feature]['distribution'].append(value)
                
                # 更新最大最小值
                stats['features'][feature]['min'] = min(
                    stats['features'][feature]['min'], 
                    value
                )
                stats['features'][feature]['max'] = max(
                    stats['features'][feature]['max'], 
                    value
                )
            else:
                # 如果特征不存在，记录警告
                import warnings
                warnings.warn(f"特征 {feature} 未在图片 {image_name} 的uncertainty中找到")
    
    # 计算统计信息
    for feature in features:
        if stats['features'][feature]['distribution']:  # 确保分布不为空
            values = np.array(stats['features'][feature]['distribution'])
            stats['features'][feature]['mean'] = float(np.mean(values))
            stats['features'][feature]['std'] = float(np.std(values))
        else:
            # 如果分布为空，设置默认值
            stats['features'][feature]['mean'] = 0.0
            stats['features'][feature]['std'] = 0.0
            stats['features'][feature]['min'] = 0.0
            stats['features'][feature]['max'] = 0.0
    
    return stats
# 默认选择器，解决的是默认方法中没有生成报告的问题  
class DefaultSelector:
    def __init__(self, uncertainty_metric):
        self.uncertainty_metric = uncertainty_metric
        
    def get_selection_report(self, selected_samples: List[str], unlabeled_results: Dict) -> Dict:
        """生成默认选择器的选择报告"""
        report = {
            'method': 'Default uncertainty-based selection',
            'metric': self.uncertainty_metric,
            'selected_samples': {},
            'statistics': {
                'total_selected': len(selected_samples),
                'uncertainty_scores': {}
            }
        }
        
        # 收集选中样本的不确定性分数
        uncertainty_scores = []
        for img_id in selected_samples:
            if img_id in unlabeled_results:
                score = unlabeled_results[img_id]['uncertainty'].get(self.uncertainty_metric, 0.0)
                report['selected_samples'][img_id] = {
                    'uncertainty_score': score
                }
                uncertainty_scores.append(score)
        
        # 计算统计信息
        if uncertainty_scores:
            report['statistics']['uncertainty_scores'] = {
                'mean': float(np.mean(uncertainty_scores)),
                'std': float(np.std(uncertainty_scores)),
                'min': float(np.min(uncertainty_scores)),
                'max': float(np.max(uncertainty_scores)),
                'median': float(np.median(uncertainty_scores))
            }
        
        return report



# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='主动学习训练')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--work-dir', help='工作目录')
    parser.add_argument('--cfg-options',nargs='+',action=DictAction,help='覆盖配置文件中的选项')
    args = parser.parse_args()
    return args

# 主函数
def main():
    args = parse_args()

    # 加载配置
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    logger = MMLogger.get_current_instance()
    
    # 打印数据集路径信息
    print(f"数据根目录: {cfg.data_root}")
    print(f"训练集图片目录: {cfg.train_dataloader.dataset.data_prefix['img']}")
    print(f"训练集标注文件: {cfg.train_dataloader.dataset.ann_file}")
    
    # 检查文件是否存在
    img_dir = cfg.train_dataloader.dataset.data_prefix['img']
    if not osp.exists(img_dir):
        print(f"警告: 图片目录不存在: {img_dir}")
    
    ann_file = cfg.train_dataloader.dataset.ann_file
    if not osp.exists(ann_file):
        print(f"警告: 标注文件不存在: {ann_file}")
    
    # 设置工作目录
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = Path('./work_dirs') / Path(args.config).stem
        
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取主动学习配置
    al_cfg = cfg.active_learning
    
    # 初始化性能跟踪
    performance_history = {
        'round': [],
        'labeled_ratio': [],        # 标注比例
        'labeled_images': [],       # 已标注图片数
        'unlabeled_images': [],     # 未标注图片数
        'total_images': [],         # 总图片数
        'labeled_annotations': [],   # 已标注标注框数量
        'val_bbox_mAP': [],        # 验证集性能
        'val_bbox_mAP_50': [],
        'val_bbox_mAP_75': [],
        'timestamp': []             # 时间戳
    }
    
    # 主动学习循环
    for active_learning_round in range(1, al_cfg.max_iterations + 1):
        print(f"\n开始第 {active_learning_round}/{al_cfg.max_iterations} 轮主动学习...")

        # 调整阈值参数      
        cfg.score_threshold, cfg.max_boxes_per_img, cfg.nms_iou_threshold = adjust_thresholds(
            cfg, active_learning_round-1, al_cfg.max_iterations
        )

        # 创建当前迭代的工作目录
        iter_work_dir = work_dir / f"round_{active_learning_round}"
        iter_work_dir.mkdir(exist_ok=True)
        
        # 更新配置中的工作目录
        cfg.work_dir = str(iter_work_dir)
        
        # 如果不是第一轮，加载上一轮的最佳模型
        if active_learning_round > 1:
            prev_iter_dir = work_dir / f"round_{active_learning_round - 1}"
            prev_ckpt = find_best_checkpoint(prev_iter_dir, logger)
            if prev_ckpt:
                logger.info(f"加载上一轮检查点: {prev_ckpt}")
                cfg.load_from = prev_ckpt
            else:
                logger.warning(f"未找到上一轮检查点")
        
        # 1. 训练学生模型
        runner = Runner.from_cfg(cfg)

        runner.train()
        
        # 2. 评估模型性能
        eval_results = {}
        try:
            # 验证集评估
            if hasattr(cfg, 'val_dataloader') and hasattr(cfg, 'val_evaluator'):
                val_results = runner.val()
                # 打印原始结果以便调试
                # logger.info(f"验证集原始结果: {val_results}")  # 已经存在的输出，是由mmdet mmengine给出的验证机评估结果，不需要在进行打印
                # 确保获取到正确的指标
                if isinstance(val_results, dict):
                    val_metrics = val_results.get('coco/bbox_mAP', 0.0)
                    val_metrics_50 = val_results.get('coco/bbox_mAP_50', 0.0)
                    val_metrics_75 = val_results.get('coco/bbox_mAP_75', 0.0)
                    val_metrics_95 = val_results.get('coco/bbox_mAP_95', 0.0)
                else:
                    val_metrics = val_metrics_50 = val_metrics_75 = 0.0
                eval_results['val'] = {
                    'bbox_mAP': val_metrics,
                    'bbox_mAP_50': val_metrics_50,
                    'bbox_mAP_75': val_metrics_75,
                    'bbox_mAP_95': val_metrics_95
                }
                # 使得输出结果更简洁  
                formatted_result = ", ".join([f"{key}: {value}" for key, value in eval_results['val'].items()])
                logger.info(f"验证集评估结果: {formatted_result}")   # 验证集评估结果: bbox_mAP: 0.001, bbox_mAP_50: 0.008, bbox_mAP_75: 0.0  # 新添加  bbox_mAP_95
        except Exception as e:
            logger.warning(f"评估过程出错: {e}")
            eval_results = {'val': {}}
        
        # 3. 使用训练好的模型进行推理   
        latest_ckpt = find_best_checkpoint(iter_work_dir, logger)
        if not latest_ckpt:
            raise FileNotFoundError(f"在 {iter_work_dir} 中未找到有效的检查点文件")
        
        # logger.info("开始计算训练数据SSC分数...")
        logger.info("Start calculating SSC scores for training dataset...")
        # 计算训练集的SSC分数
        train_pool_calculator = SSCCalculator(
            data_cfg=al_cfg.train_pool_cfg  # 直接传入整个配置字典
        )

        # 计算数据集统计信息
        train_stats = train_pool_calculator.calculate_dataset_stats()

        # 标准化训练集统计信息
        train_stats = normalize_train_stats(train_stats)

        # 保存训练集统计信息
        stats_path = iter_work_dir / 'train_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(train_stats, f, indent=2, ensure_ascii=False)

        # 4. 推理未标注数据 
        # logger.info("开始推理未标注数据...")
        logger.info("Start inferring unlabeled data...")
        teacher = SimDetectionInference(
            config_file=args.config,
            batch_size=4, # 这个参数还没有实现批次，需要修改simdetection,但在detection中已经实现
            checkpoint_file=latest_ckpt,
            output_dir=str(iter_work_dir / 'teacher_outputs'),
            enable_uncertainty=True,
            uncertainty_methods=al_cfg.inference_options.uncertainty_methods
        )
                
        if al_cfg.inference_options.sample_size == 0:
            logger.info("使用未标注池中的所有图片进行推理")
        else:
            logger.info(f"未标注池中随机采样：{al_cfg.inference_options.sample_size}张")

        # 推理未标注数据
        unlabeled_pool_results = teacher.inference(
            str(Path(al_cfg.data_root) / 'images_unlabeled'),
            sample_size= al_cfg.inference_options.sample_size
 
        )

        # 计算未标注池的uncertainty 未进行归一化
        unlabeled_pool_results_uncertainty = teacher.compute_uncertainty(
            unlabeled_pool_results,
            score_thr=al_cfg.inference_options['score_thr']
        )

        # 重新计算树冠控制系数 ，因为在计算uncertainty时，树冠控制系数无法获取到这个批次的统计信息，
        # 所以需要重新计算，并更新到uncertainty中 ，后续思考发现如果能够在UncertaintyMetrics类中，
        # 计算uncertainty时，能够获取到这个批次的统计信息，那么就不需要重新计算，并更新到uncertainty中
        unlabeled_pool_results_uncertainty = recalculate_crown_count_scores(
            unlabeled_pool_results, 
            unlabeled_pool_results_uncertainty
        )
        # 遍历未标注池的不确定性结果  ！！degub console 中运行 ！！
        # for img_name, img_data in unlabeled_pool_results_uncertainty.items():
        #     if 'uncertainty' in img_data and 'crown_count_score' in img_data['uncertainty']:
        #         crown_count_score = img_data['uncertainty']['crown_count_score']
        #         print(f"图片: {img_name}, 树冠控制系数: {crown_count_score}")
        #     else:
        #         print(f"图片: {img_name}, 没有树冠控制系数")

        # 标准化不确定性分数
        unlabeled_pool_results_uncertainty = normalize_uncertainty_scores(unlabeled_pool_results_uncertainty)


        # 定义主动学习数据集类
        dataset = ActiveCocoDataset(
            data_root=al_cfg.data_root,
            ann_file=cfg.train_dataloader.dataset.ann_file,
            data_prefix=cfg.train_dataloader.dataset.data_prefix
        )

        logger.info("Starting to select new samples...")
        logger.info("Sample selection parameters:")
        logger.info(f"  - Number of samples: {al_cfg.sample_selection['num_samples']}")
        logger.info(f"  - Uncertainty metric: {al_cfg.sample_selection['uncertainty_metric']}")
        logger.info(f"  - Sample selector: {al_cfg.sample_selection['sample_selector']}")
        logger.info(f"  - RL metric: {al_cfg.sample_selection.get('rl_metric', 'None')}")



        try:
            # 1. 获取样本选择方法
            sample_selector = al_cfg.sample_selection['sample_selector']
            selected_samples = None
            selector = None
            
            # 2. 根据选择器类型选择样本
            if sample_selector == 'default':
                # 创建默认选择器

                logger.info("Using default method to select samples")
                # 创建选择器实例
                selector = DefaultSelector(
                    uncertainty_metric=al_cfg.sample_selection['uncertainty_metric']
                )
                        # 选择样本
                selected_samples = dataset.select_samples(
                    results=unlabeled_pool_results_uncertainty,
                    **al_cfg.sample_selection
                )
                # 生成报告
                report = selector.get_selection_report(
                    selected_samples, 
                    unlabeled_pool_results_uncertainty
                )
        
                logger.info(f"Default sample selection completed - {len(selected_samples)} samples chosen")
                logger.debug(f"Selection report: {report}")

            else:
                # 3. Wasserstein或RL方法的通用处理
                # 转化未标注数据为统一格式
                unlabeled_stats = convert_unlabeled_to_stats_format(unlabeled_pool_results_uncertainty)
                
                # 保存未标注池的统计信息
                unlabeled_stats_path = iter_work_dir / 'unlabeled_stats.json'
                with open(unlabeled_stats_path, 'w', encoding='utf-8') as f:
                    json.dump(unlabeled_stats, f, indent=2, ensure_ascii=False)

                # 4. 根据选择器类型创建选择器
                if sample_selector == 'Wasserstein':
                    selector = WassersteinSampleSelector(train_stats)
                    logger.info("Using Wasserstein sample selector")

                    
                    selected_samples = selector.select_samples(
                        unlabeled_stats, 
                        select_num=al_cfg.sample_selection['num_samples']
                    )
                
                elif sample_selector == 'RL':
                    # 获取RL参数
                    q_table_path = os.path.join(str(iter_work_dir), 'rl_q_table.json')
                    selector = RLSampleSelector(
                        train_stats, 
                        metric=al_cfg.sample_selection.get('rl_metric', 'wasserstein')
                    )

                    logger.info(f"Using RL sample selector (metric: {al_cfg.sample_selection.get('rl_metric', 'wasserstein')})")
                    
                    selected_samples = selector.select_samples(
                        unlabeled_stats, 
                        select_num=al_cfg.sample_selection['num_samples'],
                        q_table_path=q_table_path
                    )
                
                
                elif sample_selector == 'Combinatorial':

                    # 获取组合优化配置，如果不存在则使用默认值
                    combo_cfg = al_cfg.sample_selection.get('combinatorial', {})
                    
                    # 设置默认权重
                    default_weights = {
                        'occlusion_score': 0.3,
                        'crown_count_score': 0.25,
                        'diversity_score': 0.2,
                        'area_var_score': 0.15,
                        'density_var_score': 0.1
                    }
                    
                    # 创建选择器实例，使用默认值
                    selector = CombinatorialSelector(
                        feature_weights=combo_cfg.get('feature_weights', default_weights),
                        ssc_weight=combo_cfg.get('ssc_weight', 0.7)
                    )
                    
                    logger.info("Using Combinatorial sample selector")
                    logger.info(f"Feature weights: {selector.feature_weights}")
                    logger.info(f"SSC weight: {selector.ssc_weight}")
                    
                    # 选择样本
                    selected_samples, report = selector.select_samples(
                        train_stats=train_stats,
                        unlabeled_stats=unlabeled_stats,
                        select_num=al_cfg.sample_selection['num_samples']
                    )
                    # 优化输出格式
                    logger.info(f"\n{'='*20} Combinatorial Selection Report {'='*20}")
                    logger.info(f"Selected samples count: {report['selected_count']}")
                    logger.info(f"Average SSC score: {report['avg_ssc']:.4f}")
                    logger.info("\nFeature Changes:")
                    logger.info(f"{'Feature':<20} {'Change':<15} {'Change %':<15}")
                    logger.info("-" * 50)
                    for metric, data in report['feature_changes'].items():
                        logger.info(
                            f"{metric:<20} {data['change']:>+.4f}      {data['change_pct']:>+.2f}%"
                        )
                    logger.info("=" * 65)

                    # 如果有优化状态信息，也输出
                    if 'optimization_status' in report:
                        logger.info(f"Optimization status: {report['optimization_status']}")

                        # 创建报告保存目录
                    report_dir = os.path.join(str(iter_work_dir), 'selection_reports')
                    os.makedirs(report_dir, exist_ok=True)

                    # 生成时间戳
                    import time
                    timestamp = time.strftime('%Y%m%d_%H%M%S')

                    # 保存JSON格式报告
                    json_report = {-
                        'timestamp': timestamp,
                        'selection_method': 'Combinatorial',
                        'selected_count': report['selected_count'],
                        'avg_ssc': float(report['avg_ssc']),
                        'feature_changes': report['feature_changes'],
                        'selected_samples': selected_samples,
                        'optimization_status': report.get('optimization_status', None),
                        'configuration': {
                            'feature_weights': selector.feature_weights,
                                'ssc_weight': selector.ssc_weight
                            }
                        }
                        
                    json_path = os.path.join(report_dir, f'selection_report_{timestamp}.json')
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_report, f, indent=2, ensure_ascii=False)

                    

                else:
                    raise ValueError(f"Unknown sample selection method: {sample_selector}")

                # 5. 生成选择报告
                report = selector.get_selection_report(selected_samples, unlabeled_stats)
                logger.info(f"{sample_selector} sample selection completed - {len(selected_samples)} samples chosen")
                logger.debug(f"Selection report: {report}")

            # 确保样本被选择
            if selected_samples is None or len(selected_samples) == 0:
                raise ValueError("No samples selected")

        except Exception as e:
            logger.error(f"Sample selection process error: {str(e)}")
            logger.warning("Attempting to use basic default method as a fallback")
            try:
                selected_samples = dataset.select_samples(
                    results=unlabeled_pool_results_uncertainty,
                    num_samples=al_cfg.sample_selection['num_samples'],
                    uncertainty_metric=al_cfg.sample_selection['uncertainty_metric']
                )
                if not selected_samples:
                    raise RuntimeError("Fallback method also failed to select samples")
                logger.info(f"Fallback method completed - {len(selected_samples)} samples chosen")
            except Exception as backup_error:
                logger.error(f"Fallback method also failed: {str(backup_error)}")
                raise RuntimeError("Sample selection failed completely, cannot continue active learning process")

        
        # 6. 更新数据集
        success = dataset.update_dataset(selected_samples)
        if not success:
            logger.error("Dataset update failed")
            raise RuntimeError("Dataset update failed")
        logger.info("Dataset updated successfully")
        
        # 7. 更新性能历史
        current_stats = dataset.get_dataset_stats()
        performance_history['round'].append(active_learning_round)
        performance_history['labeled_ratio'].append(current_stats['labeled_ratio'])
        performance_history['labeled_images'].append(current_stats['labeled_images'])
        performance_history['unlabeled_images'].append(current_stats['unlabeled_images'])
        performance_history['total_images'].append(current_stats['total_images'])
        performance_history['labeled_annotations'].append(current_stats['labeled_annotations'])
        
        # 添加验证集性能
        val_results = eval_results.get('val', {})
        performance_history['val_bbox_mAP'].append(val_results.get('bbox_mAP', 0.0))
        performance_history['val_bbox_mAP_50'].append(val_results.get('bbox_mAP_50', 0.0))
        performance_history['val_bbox_mAP_75'].append(val_results.get('bbox_mAP_75', 0.0))
        
        # 添加时间戳
        performance_history['timestamp'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # 8. 保存统计信息
        stats_info = {
            'iteration': active_learning_round,
            'selected_samples': selected_samples,
            'dataset_stats': current_stats,
            'evaluation_results': eval_results
        }
            
        with open(iter_work_dir / 'stats.json', 'w') as f:
            json.dump(stats_info, f, indent=2)
        
        # 9. 保存性能历史到CSV
        df = pd.DataFrame(performance_history)
        df.to_csv(work_dir / 'performance_history.csv', index=False)
        
        # 10. 打印当前轮次的详细信息
        print(f"\n第 {active_learning_round} 轮统计信息:")
        print(f"数据集统计:")
        print(f"  - 已标注图片数: {current_stats['labeled_images']}")
        print(f"  - 未标注图片数: {current_stats['unlabeled_images']}")
        print(f"  - 总图片数: {current_stats['total_images']}")
        print(f"  - 标注比例: {current_stats['labeled_ratio']:.2%}")
        print(f"  - 已标注框数量: {current_stats['labeled_annotations']}")
        
        if val_results:
            print(f"验证集性能:")
            print(f"  - bbox_mAP: {val_results.get('bbox_mAP', 0.0):.4f}")
            print(f"  - bbox_mAP_50: {val_results.get('bbox_mAP_50', 0.0):.4f}")
            print(f"  - bbox_mAP_75: {val_results.get('bbox_mAP_75', 0.0):.4f}")
        
        # 清理 GPU 内存
        torch.cuda.empty_cache()

if __name__ == '__main__':
    import sys
    # sys.argv = ['sual/minal_train.py', 'configs/al_config/al_faster-rcnn_sscmin.py', '--work-dir', 'work_dirs/al_ssc']
    #sys.argv = ['sual/minal_train.py', 'custom_config/faster-rcnn_ssc_Default.py', '--work-dir', 'work_dirs/al_faster-rcnn_ssc_default']
    # sys.argv = ['sual/minal_train.py', 'custom_config/faster-rcnn_ssc_combinatorial.py', '--work-dir', 'work_dirs/al_faster-rcnn_ssc_combinatorial']
    
    # sys.argv = ['sual/minal_train.py', 'custom_config/faster-rcnn_rl.py', '--work-dir', 'work_dirs/al_faster-rcnn_rl']
    # 随即不确定性加默认筛选策略
    sys.argv =['sual/minal_train.py',"custom_config/al_faster-rcnn_random.py",'--work-dir', 'work_dirs/al_faster-rcnn_basic_default']
    main()