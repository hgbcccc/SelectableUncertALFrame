
# import sys
# # sys.path.append('E:\\sual')s
# import sys
# import locale
# sys.stdout.reconfigure(encoding='utf-8')

import argparse
from pathlib import Path
import json
import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from sual.inference.detector import DetectionInference
from sual.core.datasets import ActiveCocoDataset
import os.path as osp
import re
from typing import Optional, List
from mmengine.logging import MMLogger
from sual.core.hooks import ActiveLearningEvalHook
from datetime import datetime
import pandas as pd

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
                logger.info(f"验证集原始结果: {val_results}")
                # 确保获取到正确的指标
                if isinstance(val_results, dict):
                    val_metrics = val_results.get('coco/bbox_mAP', 0.0)
                    val_metrics_50 = val_results.get('coco/bbox_mAP_50', 0.0)
                    val_metrics_75 = val_results.get('coco/bbox_mAP_75', 0.0)
                else:
                    val_metrics = val_metrics_50 = val_metrics_75 = 0.0
                eval_results['val'] = {
                    'bbox_mAP': val_metrics,
                    'bbox_mAP_50': val_metrics_50,
                    'bbox_mAP_75': val_metrics_75
                }
                logger.info(f"验证集评估结果: {eval_results['val']}")
        except Exception as e:
            logger.warning(f"评估过程出错: {e}")
            eval_results = {'val': {}}
        
        # 3. 使用训练好的模型进行推理
        latest_ckpt = find_best_checkpoint(iter_work_dir, logger)
        if not latest_ckpt:
            raise FileNotFoundError(f"在 {iter_work_dir} 中未找到有效的检查点文件")
            
        logger.info("开始推理未标注数据...")
        teacher = DetectionInference(
            config_file=args.config,
            batch_size=4,
            checkpoint_file=latest_ckpt,
            output_dir=str(iter_work_dir / 'teacher_outputs'),
            enable_uncertainty=True
        )
        
        # 4. 推理未标注数据
        results = teacher.inference(
            str(Path(al_cfg.data_root) / 'images_unlabeled'),
            **al_cfg.inference_options
            # **al_cfg.inference_options.selected_metric
            #   
        )
#################################### result 的结果###########################################
#         '01c71fc3-dea0-429b-b703-f47c4c3e2bbb.jpg': {
#     'result': <DetDataSample(
#         META INFORMATION
#         img_id: 0
#         ori_shape: (1536, 1536)
#         pad_shape: (512, 512)
#         img_shape: (512, 512)
#         scale_factor: (0.3333333333333333, 0.3333333333333333)
#         batch_input_shape: (512, 512)
#         img_path: None

#         DATA FIELDS
#         gt_instances: <InstanceData(...>
#         ignored_instances: <InstanceData(...>
#         pred_instances: <InstanceData(
#             DATA FIELDS
#             scores: tensor([...])
#             bboxes: tensor([...])
#             labels: tensor([...])
#         )>
#     )>,
#     'vis_path': 'work_dirs/faster-rcnn/al_entropy/round_1/teacher_outputs/20250223_023806/visualize/01c71fc3-dea0-429b-b703-f47c4c3e2bbb_vis.jpg',
#     'uncertainty': {
#         'entropy': 5.701963424682617,
#         'normalized_entropy': 0.9996810800583649
#     }
# }
        logger.info("开始选择新样本...")
        logger.info(f"sample_selection 参数: {al_cfg.sample_selection}")
        
        # 5. 选择新样本
        dataset = ActiveCocoDataset(
            data_root=al_cfg.data_root,
            ann_file=cfg.train_dataloader.dataset.ann_file,
            data_prefix=cfg.train_dataloader.dataset.data_prefix
        )
        
        selected_samples = dataset.select_samples(
            results=results,
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
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
# 带有验证集测试评估的主动学习流程，导致训练过程很慢，但是便于记录
# def main():
#     args = parse_args()
    
#     # 加载配置
#     cfg = Config.fromfile(args.config)
#     if args.cfg_options is not None:
#         cfg.merge_from_dict(args.cfg_options)
    
#     logger = MMLogger.get_current_instance()
    
#     # 打印数据集路径信息
#     print(f"数据根目录: {cfg.data_root}")
#     print(f"训练集图片目录: {cfg.train_dataloader.dataset.data_prefix['img']}")
#     print(f"训练集标注文件: {cfg.train_dataloader.dataset.ann_file}")
    
#     # 检查文件是否存在
#     img_dir = cfg.train_dataloader.dataset.data_prefix['img']
#     if not osp.exists(img_dir):
#         print(f"警告: 图片目录不存在: {img_dir}")
    
#     ann_file = cfg.train_dataloader.dataset.ann_file
#     if not osp.exists(ann_file):
#         print(f"警告: 标注文件不存在: {ann_file}")
    
#     # 设置工作目录
#     if args.work_dir is not None:
#         cfg.work_dir = args.work_dir
#     elif cfg.get('work_dir', None) is None:
#         cfg.work_dir = Path('./work_dirs') / Path(args.config).stem
        
#     work_dir = Path(cfg.work_dir)
#     work_dir.mkdir(parents=True, exist_ok=True)
    
#     # 获取主动学习配置
#     al_cfg = cfg.active_learning
    
#     # 初始化性能跟踪
#     performance_history = {
#         'round': [],
#         'labeled_ratio': [],        # 标注比例
#         'labeled_images': [],       # 已标注图片数
#         'unlabeled_images': [],     # 未标注图片数
#         'total_images': [],         # 总图片数
#         'labeled_annotations': [],   # 已标注标注框数量
#         'val_bbox_mAP': [],        # 验证集性能
#         'val_bbox_mAP_50': [],
#         'val_bbox_mAP_75': [],
#         'test_bbox_mAP': [],       # 测试集性能
#         'test_bbox_mAP_50': [],
#         'test_bbox_mAP_75': [],
#         'timestamp': []
#     }
    
#     # 主动学习循环
#     for active_learning_round in range(1, al_cfg.max_iterations + 1):
#         print(f"\n开始第 {active_learning_round}/{al_cfg.max_iterations} 轮主动学习...")
        
#         # 创建当前迭代的工作目录
#         iter_work_dir = work_dir / f"round_{active_learning_round}"
#         iter_work_dir.mkdir(exist_ok=True)
        
#         # 更新配置中的工作目录
#         cfg.work_dir = str(iter_work_dir)
        
#         # 如果不是第一轮，加载上一轮的最佳模型
#         if active_learning_round > 1:
#             prev_iter_dir = work_dir / f"round_{active_learning_round - 1}"
#             prev_ckpt = find_best_checkpoint(prev_iter_dir, logger)
#             if prev_ckpt:
#                 logger.info(f"加载上一轮检查点: {prev_ckpt}")
#                 cfg.load_from = prev_ckpt
#             else:
#                 logger.warning(f"未找到上一轮检查点")
        
#         # 1. 训练学生模型
#         runner = Runner.from_cfg(cfg)
#         runner.train()
        
#         # 2. 评估模型性能
#         eval_results = {}
#         try:
#             # 验证集评估
#             if hasattr(cfg, 'val_dataloader') and hasattr(cfg, 'val_evaluator'):
#                 val_results = runner.val()
#                 # 打印原始结果以便调试
#                 logger.info(f"验证集原始结果: {val_results}")
#                 # 确保获取到正确的指标
#                 if isinstance(val_results, dict):
#                     val_metrics = val_results.get('coco/bbox_mAP', 0.0)
#                     val_metrics_50 = val_results.get('coco/bbox_mAP_50', 0.0)
#                     val_metrics_75 = val_results.get('coco/bbox_mAP_75', 0.0)
#                 else:
#                     val_metrics = val_metrics_50 = val_metrics_75 = 0.0
#                 eval_results['val'] = {
#                     'bbox_mAP': val_metrics,
#                     'bbox_mAP_50': val_metrics_50,
#                     'bbox_mAP_75': val_metrics_75
#                 }
#                 logger.info(f"验证集评估结果: {eval_results['val']}")
            
#             # # 测试集评估
#             # if hasattr(cfg, 'test_dataloader') and hasattr(cfg, 'test_evaluator'):
#             #     test_results = runner.test()
#             #     # 打印原始结果以便调试
#             #     logger.info(f"测试集原始结果: {test_results}")
#             #     # 确保获取到正确的指标
#             #     if isinstance(test_results, dict):
#             #         test_metrics = test_results.get('coco/bbox_mAP', 0.0)
#             #         test_metrics_50 = test_results.get('coco/bbox_mAP_50', 0.0)
#             #         test_metrics_75 = test_results.get('coco/bbox_mAP_75', 0.0)
#             #     else:
#             #         test_metrics = test_metrics_50 = test_metrics_75 = 0.0
#             #     eval_results['test'] = {
#             #         'bbox_mAP': test_metrics,
#             #         'bbox_mAP_50': test_metrics_50,
#             #         'bbox_mAP_75': test_metrics_75
#             #     }
#             #     logger.info(f"测试集评估结果: {eval_results['test']}")
#         except Exception as e:
#             logger.warning(f"评估过程出错: {e}")
#             eval_results = {'val': {}, 'test': {}}
        
#         # 3. 使用训练好的模型进行推理
#         latest_ckpt = find_best_checkpoint(iter_work_dir, logger)
#         if not latest_ckpt:
#             raise FileNotFoundError(f"在 {iter_work_dir} 中未找到有效的检查点文件")
            
#         logger.info("开始推理未标注数据...")
#         teacher = DetectionInference(
#             config_file=args.config,
#             batch_size=4,
#             checkpoint_file=latest_ckpt,
#             output_dir=str(iter_work_dir / 'teacher_outputs'),
#             enable_uncertainty=True
#         )
        
#         # 4. 推理未标注数据
#         results = teacher.inference(
#             str(Path(al_cfg.data_root) / 'images_unlabeled'),
#             **al_cfg.inference_options
#         )
#         logger.info("开始选择新样本...")
#         logger.info(f"sample_selection 参数: {al_cfg.sample_selection}")
        
#         # 5. 选择新样本
#         dataset = ActiveCocoDataset(
#             data_root=al_cfg.data_root,
#             ann_file=cfg.train_dataloader.dataset.ann_file,
#             data_prefix=cfg.train_dataloader.dataset.data_prefix
#         )
        
#         selected_samples = dataset.select_samples(
#             results=results,
#             **al_cfg.sample_selection
#         )
#         logger.info(f"选择完成，选中样本数量: {len(selected_samples)}")
#         logger.info("开始更新数据集...")
#         # 6. 更新数据集
#         success = dataset.update_dataset(selected_samples)
#         if not success:
#             logger.error("数据集更新失败")
#             raise RuntimeError("数据集更新失败")

#         logger.info("数据集更新成功")
        
#         # 7. 更新性能历史
#         current_stats = dataset.get_dataset_stats()
#         performance_history['round'].append(active_learning_round)
#         performance_history['labeled_ratio'].append(current_stats['labeled_ratio'])
#         performance_history['labeled_images'].append(current_stats['labeled_images'])
#         performance_history['unlabeled_images'].append(current_stats['unlabeled_images'])
#         performance_history['total_images'].append(current_stats['total_images'])
#         performance_history['labeled_annotations'].append(current_stats['labeled_annotations'])
        
#         # 添加验证集性能
#         val_results = eval_results.get('val', {})
#         performance_history['val_bbox_mAP'].append(val_results.get('bbox_mAP', 0.0))
#         performance_history['val_bbox_mAP_50'].append(val_results.get('bbox_mAP_50', 0.0))
#         performance_history['val_bbox_mAP_75'].append(val_results.get('bbox_mAP_75', 0.0))
        
#         # 添加测试集性能
# #         test_results = eval_results.get('test', {})
# #         performance_history['test_bbox_mAP'].append(test_results.get('bbox_mAP', 0.0))
# #         performance_history['test_bbox_mAP_50'].append(test_results.get('bbox_mAP_50', 0.0))
# #         performance_history['test_bbox_mAP_75'].append(test_results.get('bbox_mAP_75', 0.0))
        
# #         performance_history['timestamp'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
#         # 8. 保存统计信息
#         stats_info = {
#             'iteration': active_learning_round,
#             'selected_samples': selected_samples,
#             'dataset_stats': current_stats,
#             'evaluation_results': eval_results
#         }
        
#         with open(iter_work_dir / 'stats.json', 'w') as f:
#             json.dump(stats_info, f, indent=2)
        
#         # 9. 保存性能历史到CSV
#         df = pd.DataFrame(performance_history)
#         df.to_csv(work_dir / 'performance_history.csv', index=False)
        
#         # 10. 打印当前轮次的详细信息
#         print(f"\n第 {active_learning_round} 轮统计信息:")
#         print(f"数据集统计:")
#         print(f"  - 已标注图片数: {current_stats['labeled_images']}")
#         print(f"  - 未标注图片数: {current_stats['unlabeled_images']}")
#         print(f"  - 总图片数: {current_stats['total_images']}")
#         print(f"  - 标注比例: {current_stats['labeled_ratio']:.2%}")
#         print(f"  - 已标注框数量: {current_stats['labeled_annotations']}")
        
#         if val_results:
#             print(f"验证集性能:")
#             print(f"  - bbox_mAP: {val_results.get('bbox_mAP', 0.0):.4f}")
#             print(f"  - bbox_mAP_50: {val_results.get('bbox_mAP_50', 0.0):.4f}")
#             print(f"  - bbox_mAP_75: {val_results.get('bbox_mAP_75', 0.0):.4f}")
        
#         # if test_results:
#         #     print(f"测试集性能:")
#         #     print(f"  - bbox_mAP: {test_results.get('bbox_mAP', 0.0):.4f}")
#         #     print(f"  - bbox_mAP_50: {test_results.get('bbox_mAP_50', 0.0):.4f}")
#         #     print(f"  - bbox_mAP_75: {test_results.get('bbox_mAP_75', 0.0):.4f}")
        
#         # 绘制性能曲线
# #         try:
# #             import matplotlib.pyplot as plt
# #             plt.style.use('seaborn')
            
# #             # 创建图表
# #             fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
# #             # 标注进度图
# #             ax1.plot(df['round'], df['labeled_images'], 'b-', label='已标注图片', linewidth=2)
# #             ax1.plot(df['round'], df['unlabeled_images'], 'r--', label='未标注图片', linewidth=2)
# #             ax1.set_xlabel('主动学习轮次')
# #             ax1.set_ylabel('图片数量')
# #             ax1.set_title('数据集标注进度')
# #             ax1.legend()
# #             ax1.grid(True)
            
# #             # 验证集性能-轮次曲线
# #             if val_results:
# #                 ax2.plot(df['round'], df['val_bbox_mAP'], 'b-', label='val_mAP', linewidth=2)
# #                 ax2.plot(df['round'], df['val_bbox_mAP_50'], 'g--', label='val_mAP_50', linewidth=2)
# #                 ax2.plot(df['round'], df['val_bbox_mAP_75'], 'r--', label='val_mAP_75', linewidth=2)
# #                 ax2.set_xlabel('主动学习轮次')
# #                 ax2.set_ylabel('验证集性能')
# #                 ax2.set_title('验证集性能-轮次曲线')
# #                 ax2.legend()
# #                 ax2.grid(True)
            
# #             # 测试集性能-轮次曲线
# #             if test_results:
# #                 ax3.plot(df['round'], df['test_bbox_mAP'], 'b-', label='test_mAP', linewidth=2)
# #                 ax3.plot(df['round'], df['test_bbox_mAP_50'], 'g--', label='test_mAP_50', linewidth=2)
# #                 ax3.plot(df['round'], df['test_bbox_mAP_75'], 'r--', label='test_mAP_75', linewidth=2)
# #                 ax3.set_xlabel('主动学习轮次')
# #                 ax3.set_ylabel('测试集性能')
# #                 ax3.set_title('测试集性能-轮次曲线')
# #                 ax3.legend()
# #                 ax3.grid(True)
            
# #             # 性能-标注比例曲线
# #             if test_results:
# #                 ax4.plot(df['labeled_ratio'], df['test_bbox_mAP'], 'b-', label='test_mAP', linewidth=2)
# #                 ax4.plot(df['labeled_ratio'], df['test_bbox_mAP_50'], 'g--', label='test_mAP_50', linewidth=2)
# #                 ax4.plot(df['labeled_ratio'], df['test_bbox_mAP_75'], 'r--', label='test_mAP_75', linewidth=2)
# #                 ax4.set_xlabel('标注数据占比')
# #                 ax4.set_ylabel('测试集性能')
# #                 ax4.set_title('性能-标注比例曲线')
# #                 ax4.legend()
# #                 ax4.grid(True)
            
# #             plt.tight_layout()
# #             plt.savefig(work_dir / 'performance_curves.png', dpi=300, bbox_inches='tight')
# #             plt.close()
            
# #         except Exception as e:
# #             logger.warning(f"绘制性能曲线失败: {e}")
        
#         # 清理 GPU 内存
#         torch.cuda.empty_cache()
# if __name__ == '__main__':
    
#     main()