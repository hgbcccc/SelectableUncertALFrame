

import argparse
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
from sual.core.hooks import ActiveLearningEvalHook
from datetime import datetime
import pandas as pd
from scipy.stats import wasserstein_distance
from typing import Dict, List, Optional, Tuple


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


def parse_args():
    parser = argparse.ArgumentParser(description='主动学习训练')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--work-dir', help='工作目录')
    parser.add_argument('--cfg-options',nargs='+',action=DictAction,help='覆盖配置文件中的选项')
    args = parser.parse_args()
    return args


def main():
    global train_results, train_uncertainty , results ,results_un
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
        
        # 3. 使用训练好的模型进行推理   # 后续应该需要直接使用是训练集中GT来计算，不是使用模型推理的结果
        latest_ckpt = find_best_checkpoint(iter_work_dir, logger)
        if not latest_ckpt:
            raise FileNotFoundError(f"在 {iter_work_dir} 中未找到有效的检查点文件")
                 
        # 4. 推理未标注数据 
        logger.info("开始推理未标注数据...")
        teacher = SimDetectionInference(
            config_file=args.config,
            batch_size=4,
            checkpoint_file=latest_ckpt,
            output_dir=str(iter_work_dir / 'teacher_outputs'),
            enable_uncertainty=True,
            uncertainty_methods=al_cfg.inference_options.uncertainty_methods
        )
        logger.info(f"未标注池中随机采样：{al_cfg.inference_options.sample_size}张")

        # 推理未标注数据
        results = teacher.inference(
            str(Path(al_cfg.data_root) / 'images_unlabeled'),
            sample_size= al_cfg.inference_options.sample_size
 
        )

        result_un = teacher.compute_uncertainty(
            results,
            score_thr=al_cfg.inference_options['score_thr']
        )
        # print(result_un)
        # 保存 result_un 到文件
        # result_un_path = iter_work_dir / 'result_un.json'
        # with open(result_un_path, 'w', encoding='utf-8') as f:
        #     json.dump(result_un, f, indent=2, ensure_ascii=False)

        # logger.info(f"result_un 已保存至: {result_un_path}")

        # 计算并打印 result_un 中的 ssc_score 平均值
        ssc_scores_result_un = []
        for img_name, info in result_un.items():
            if 'uncertainty' in info and 'ssc_score' in info['uncertainty']:
                ssc_scores_result_un.append(info['uncertainty']['ssc_score'])

        if ssc_scores_result_un:
            avg_ssc_score_result_un = np.mean(ssc_scores_result_un)
            print(f"result_un 中的 ssc_score 平均值: {avg_ssc_score_result_un:.4f}")
        else:
            print("result_un 中没有有效的 ssc_score")

        # 定义主动学习数据集类
        dataset = ActiveCocoDataset(
            data_root=al_cfg.data_root,
            ann_file=cfg.train_dataloader.dataset.ann_file,
            data_prefix=cfg.train_dataloader.dataset.data_prefix
        )
        #定义Wasserstein来计算未标注池和标注训练池的分布差距
        wasserstein_scorer = WassersteinBalancedScorer(
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            mapd_threshold=0.25
        )
        # 构建与dataset.select_samples相同格式的输入 这样可能不经过activate_dataset.py的select_samples的排序并且选择样本，
        # 直接使用WassersteinBalancedScorer的排序并选择
        processed_results = {}
        for img_name, info in result_un.items():
            if 'uncertainty' in info:
                # 计算Wasserstein平衡得分
                balanced_metrics = wasserstein_scorer.compute_balanced_score(info['uncertainty'])
                # 保持与原始uncertainty相同的结构，但更新ssc_score
                info['uncertainty']['ssc_score'] = balanced_metrics['wasserstein_balanced_score']
                processed_results[img_name] = info

        
        # 计算未标注数据的不确定性
        # 使用dataset的select_samples方法进行选择
        # 5. 选择新样本
        logger.info("开始选择新样本...")
        logger.info(f"sample_selection 参数: {al_cfg.sample_selection}")
        # print(f"processed_results: {processed_results}")
        # print(processed_results)

        # 计算并打印 processed_results 中的 ssc_score 平均值
        ssc_scores_processed_results = []
        for img_name, info in processed_results.items():
            if 'uncertainty' in info and 'ssc_score' in info['uncertainty']:
                ssc_scores_processed_results.append(info['uncertainty']['ssc_score'])

        if ssc_scores_processed_results:
            avg_ssc_score_processed_results = np.mean(ssc_scores_processed_results)
            print(f"processed_results 中的 ssc_score 平均值: {avg_ssc_score_processed_results:.4f}")
        else:
            print("processed_results 中没有有效的 ssc_score")

        selected_samples = dataset.select_samples(
            results=processed_results,
            **al_cfg.sample_selection
        )
        logger.info(f"选择完成，选中样本数量: {len(selected_samples)}")
        logger.info("开始更新数据集...")
        
        # 6. 更新数据集
        success = dataset.update_dataset(selected_samples)
        if not success:
            logger.error("数据集更新失败")
            raise RuntimeError("数据集更新失败")
        logger.info("数据集更新成功")
        
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
    sys.argv = ['sual/ori_train.py', 'configs/al_config/al_config/al_faster-rcnn_sscmin.py', '--work-dir', 'work_dirs/al_ssc']
    main()
    # main()
    # print("Global Train Results:", train_results)
    # print("Global Train Uncertainty:", train_uncertainty)
