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
from sual.utils.training_utils import find_best_checkpoint, adjust_thresholds
from sual.utils.normalization_utils import normalize_uncertainty_scores, normalize_train_stats
from sual.utils.recalculate_crown_count_scores import recalculate_crown_count_scores
from sual.utils.convert_unlabeled_to_stats_format import convert_unlabeled_to_stats_format
from sual.core.uncertainty.measures.ssc_calculator import SSCCalculator
from sual.core.balancedscorer import WassersteinSampleSelector, RLSampleSelector, CombinatorialSelector
from sual.core.balancedscorer.defaultSelector import DefaultSelector
from sual.core.uncertainty.metrics import UncertaintyMetrics, MUSCDBSampling
from tqdm import tqdm

def setup_work_dir(cfg: Config, args: argparse.Namespace) -> Path:
    """设置工作目录"""
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = Path('./work_dirs') / Path(args.config).stem
        
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir

def init_performance_history() -> Dict:
    """初始化性能跟踪字典"""
    return {
        'round': [],
        'labeled_ratio': [],
        'labeled_images': [],
        'unlabeled_images': [],
        'total_images': [],
        'labeled_annotations': [],
        'val_bbox_mAP': [],
        'val_bbox_mAP_50': [],
        'val_bbox_mAP_75': [],
        'timestamp': []
    }

def check_need_ssc(cfg: Config) -> bool:
    """检查是否需要计算SSC分数和训练集统计"""
    al_cfg = cfg.active_learning
    # 检查选择器类型
    selector_type = al_cfg.sample_selection.get('sample_selector', '')
    # 检查是否使用SSC相关度量
    selected_metric = al_cfg.inference_options.get('selected_metric', '')
    
    # 以下选择器都需要训练集统计：
    need_train_stats_selectors = {
        'Combinatorial',
        'Wasserstein',
        'RL'
    }
    
    return (
        selector_type in need_train_stats_selectors or 
        selected_metric == 'ssc_score'
    )

def check_resume_state(work_dir: Path, max_iterations: int, logger: MMLogger) -> int:
    """检查训练状态，决定从哪一轮恢复训练"""
    logger.info("检查训练恢复状态...")
    
    # 查找现有轮次目录
    round_dirs = sorted([d for d in work_dir.glob("round_*") if d.is_dir()], 
                       key=lambda x: int(str(x).split("_")[-1]))
    
    if not round_dirs:
        logger.info("未找到现有训练轮次，将从第1轮开始训练")
        return 1
    
    # 检查最后一轮是否完成
    last_round_dir = round_dirs[-1]
    last_round_num = int(str(last_round_dir).split("_")[-1])
    
    # 检查是否存在stats.json文件，该文件表示轮次已完成
    stats_file = last_round_dir / 'stats.json'
    if stats_file.exists():
        # 最后一轮已完成，从下一轮开始
        start_round = last_round_num + 1
        logger.info(f"检测到第{last_round_num}轮已完成，将从第{start_round}轮开始训练")
    else:
        # 最后一轮未完成，从该轮重新开始
        start_round = last_round_num
        logger.info(f"检测到第{last_round_num}轮未完成，将从第{start_round}轮重新开始训练")
    
    # 如果超过最大轮次，结束训练
    if start_round > max_iterations:
        logger.info(f"已完成所有{max_iterations}轮训练，无需继续")
        exit(0)
    
    return start_round

def load_performance_history(work_dir: Path, performance_history: Dict, logger: MMLogger) -> None:
    """加载性能历史记录"""
    history_csv = work_dir / 'performance_history.csv'
    if history_csv.exists():
        try:
            df = pd.read_csv(history_csv)
            for col in performance_history.keys():
                if col in df.columns:
                    performance_history[col] = df[col].tolist()
            logger.info(f"已加载现有性能历史记录，包含{len(df)}轮数据")
        except Exception as e:
            logger.warning(f"加载性能历史记录失败: {e}")

def train_model(cfg: Config, work_dir: Path, round_num: int, logger: MMLogger) -> Tuple[Runner, Dict]:
    """训练模型并评估"""
    # 调整阈值参数
    cfg.score_threshold, cfg.max_boxes_per_img, cfg.nms_iou_threshold = \
        adjust_thresholds(cfg, round_num-1, cfg.active_learning.max_iterations)
    
    # 加载上一轮模型
    if round_num > 1:
        load_previous_checkpoint(cfg, work_dir, round_num, logger)
        
    # 训练模型
    runner = Runner.from_cfg(cfg)
    runner.train()
    
    # 评估模型
    eval_results = evaluate_model(runner, logger)
    
    return runner, eval_results

def load_previous_checkpoint(cfg: Config, work_dir: Path, round_num: int, logger: MMLogger) -> None:
    """加载上一轮检查点"""
    prev_iter_dir = work_dir / f"round_{round_num - 1}"
    prev_ckpt = find_best_checkpoint(prev_iter_dir, logger)
    if prev_ckpt:
        logger.info(f"加载上一轮检查点: {prev_ckpt}")
        cfg.load_from = prev_ckpt
    else:
        logger.warning("未找到上一轮检查点")

def evaluate_model(runner: Runner, logger: MMLogger) -> Dict:
    """评估模型性能"""
    eval_results = {'val': {}}
    try:
        if hasattr(runner.cfg, 'val_dataloader') and hasattr(runner.cfg, 'val_evaluator'):
            val_results = runner.val()
            if isinstance(val_results, dict):
                eval_results['val'] = {
                    'bbox_mAP': val_results.get('coco/bbox_mAP', 0.0),
                    'bbox_mAP_50': val_results.get('coco/bbox_mAP_50', 0.0),
                    'bbox_mAP_75': val_results.get('coco/bbox_mAP_75', 0.0),
                    'bbox_mAP_95': val_results.get('coco/bbox_mAP_95', 0.0)
                }
            formatted_result = ", ".join([f"{key}: {value}" for key, value in eval_results['val'].items()])
            logger.info(f"验证集评估结果: {formatted_result}")
    except Exception as e:
        logger.warning(f"评估过程出错: {e}")
    return eval_results

def process_unlabeled_data(cfg: Config, work_dir: Path, round_num: int, logger: MMLogger) -> Tuple[Dict, Dict]:
    """处理未标注数据"""
    train_stats = {}
    al_cfg = cfg.active_learning
    selector_type = al_cfg.sample_selection['sample_selector']
    
    # 创建轮次工作目录
    iter_work_dir = work_dir / f"round_{round_num}"
    iter_work_dir.mkdir(exist_ok=True)
    
    # 检查是否需要训练集统计
    if check_need_ssc(cfg):
        logger.info(f"计算训练集统计信息 (由{selector_type}选择器或SSC度量要求)")
        train_stats = calculate_train_stats(iter_work_dir, al_cfg, logger)
    else:
        logger.debug(f"使用{selector_type}选择器，不需要训练集统计信息")
    
    # 推理未标注数据
    unlabeled_results = infer_unlabeled_data(cfg, work_dir, round_num, iter_work_dir, logger)
    
    # 转换未标注数据为统计格式并保存
    unlabeled_stats = convert_unlabeled_to_stats_format(unlabeled_results)
    
    # 标准化未标注数据统计
    unlabeled_stats = normalize_train_stats(unlabeled_stats)
    
    for feat in ['ssc_score', 'occlusion_score', 'crown_count_score', 'diversity_score', 'area_var_score', 'density_var_score']:
        mean = unlabeled_stats['features'][feat]['mean']
        distribution = unlabeled_stats['features'][feat]['distribution']
        is_unique = len(set(distribution)) == 1
        logger.info(f"{feat} mean: {mean}")
        logger.info(f"{feat} is unique: {is_unique}")
    
    # 保存未标注数据统计信息
    unlabeled_stats_path = iter_work_dir / 'unlabeled_stats.json'
    try:
        with open(unlabeled_stats_path, 'w', encoding='utf-8') as f:
            json.dump(unlabeled_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"未标注数据统计信息已保存至: {unlabeled_stats_path}")
    except Exception as e:
        logger.error(f"保存未标注数据统计信息失败: {str(e)}")
    
    return train_stats, unlabeled_results

def calculate_train_stats(iter_work_dir: Path, al_cfg: Config, logger: MMLogger) -> Dict:
    """计算训练集统计信息"""
    logger.info("开始计算训练数据SSC分数...")
    calculator = SSCCalculator(data_cfg=al_cfg.train_pool_cfg)
    train_stats = calculator.calculate_dataset_stats()
    train_stats = normalize_train_stats(train_stats)
    
    # 保存统计信息
    stats_path = iter_work_dir / 'train_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(train_stats, f, indent=2, ensure_ascii=False)
        
    return train_stats

def infer_unlabeled_data(cfg: Config, work_dir: Path, round_num: int, iter_work_dir: Path, logger: MMLogger) -> Dict:
    """推理未标注数据"""
    logger.info("开始推理未标注数据...")
    latest_ckpt = find_best_checkpoint(iter_work_dir, logger)
    if not latest_ckpt:
        raise FileNotFoundError(f"在 {iter_work_dir} 中未找到有效的检查点文件")
        
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 构建输出目录 work_dir/round_x/teacher_outputs
    teacher_output_dir = Path(work_dir) / f'round_{round_num}' / 'teacher_outputs'
    teacher_output_dir.mkdir(parents=True, exist_ok=True)

    al_cfg = cfg.active_learning
    
    teacher = SimDetectionInference(
        # config_file=cfg.,
        config_file=cfg._filename,
        # config_file=cfg.workflow[0][0],
        batch_size=al_cfg.inference_options.batch_size,
        checkpoint_file=latest_ckpt,
        output_dir=str(teacher_output_dir),
        enable_uncertainty=True,
        uncertainty_methods=al_cfg.inference_options.uncertainty_methods
    )

    # 执行推理
    results = teacher.inference(
        str(Path(al_cfg.data_root) / 'images_unlabeled'),
        sample_size=al_cfg.inference_options.sample_size
    )

    # 根据配置决定使用哪个不确定性计算器
    uncertainty_methods = al_cfg.inference_options.get('uncertainty_methods', ['all'])
    score_thr = al_cfg.inference_options.get('score_thr', 0.09)
    
    if ('mus' in uncertainty_methods and 
        al_cfg.sample_selection.get('uncertainty_metric') == 'mus'):
        # 使用两阶段MUS-CDB采样策略
        logger.info("使用MUS-CDB两阶段采样策略")
        uncertainty_calculator = MUSCDBSampling()
        use_two_stage = True
    else:
        # 使用传统的不确定性度量
        logger.info("使用单阶段不确定性度量")
        uncertainty_calculator = UncertaintyMetrics()
        use_two_stage = False

    # 计算不确定性
    processed_results = {}
    for img_name, result_info in tqdm(results.items(), desc="计算不确定性"):
        try:
            result = result_info['result']
            if use_two_stage:
                # 对于MUS-CDB策略，直接计算MUS得分
                uncertainty = {'mus': uncertainty_calculator.compute_mus(result, theta=score_thr)}
            else:
                # 传统策略，计算所有指定的不确定性度量
                uncertainty = uncertainty_calculator.compute_uncertainty(
                    result,
                    methods=uncertainty_methods,
                    min_score_thresh=score_thr
                )
            processed_results[img_name] = {
                'result': result,
                'vis_path': result_info.get('vis_path'),
                'uncertainty': uncertainty
            }
        except Exception as e:
            logger.error(f"计算不确定性时出错 ({img_name}): {str(e)}")
            processed_results[img_name] = result_info

    # 直接保存到 teacher_output_dir/uncertainty/uncertainty_results.json
    uncertainty_dir = teacher_output_dir / 'uncertainty'
    uncertainty_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_path = uncertainty_dir / 'uncertainty_results.json'

    with open(uncertainty_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'uncertainty_methods': uncertainty_methods,
            'results': {
                img_name: {
                    'uncertainty': info['uncertainty']
                } for img_name, info in processed_results.items()
                if 'uncertainty' in info
            }
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"不确定性结果已保存至: {uncertainty_path}")
    results = processed_results  # 更新结果

    # 重新计算树冠控制系数
    results = recalculate_crown_count_scores(results, results)

    # 标准化不确定性分数
    results = normalize_uncertainty_scores(results)

    return results

def create_selector(selector_type: str, train_stats: Dict, al_cfg: Config, logger: MMLogger):
    """创建样本选择器"""
    if selector_type == 'default':
        logger.info(f"使用默认选择器")
        # 检查是否使用MUS-CDB策略
        if ('mus' in al_cfg.inference_options.get('uncertainty_methods', []) and 
            al_cfg.sample_selection.get('uncertainty_metric') == 'mus'):
            logger.info("配置为MUS-CDB策略")
            return MUSCDBSampling()  # 使用MUS-CDB选择器
        else:
            return DefaultSelector(al_cfg.sample_selection['uncertainty_metric'])
    elif selector_type == 'Wasserstein':
        logger.info(f"使用Wasserstein选择器")
        return WassersteinSampleSelector(train_stats)
    elif selector_type == 'RL':
        logger.info(f"使用RL选择器")
        return RLSampleSelector(
            train_stats,
            metric=al_cfg.sample_selection.get('rl_metric', 'wasserstein')
        )
    elif selector_type == 'Combinatorial':
        logger.info(f"使用组合选择器")
        combo_cfg = al_cfg.sample_selection.get('combinatorial', {})
        return CombinatorialSelector(
            feature_weights=combo_cfg.get('feature_weights'),
            ssc_weight=combo_cfg.get('ssc_weight', 0.7)
        )
    else:
        raise ValueError(f"未知的选择器类型: {selector_type}")

def run_selector(selector, dataset, unlabeled_results, train_stats, cfg: Config, logger: MMLogger) -> List[str]:
    """运行样本选择器"""
    selector_type = cfg.active_learning.sample_selection['sample_selector']
    
    if selector_type == 'default':
        if isinstance(selector, MUSCDBSampling):
            # MUS-CDB两阶段采样策略
            logger.info("执行MUS-CDB两阶段采样")
            # 获取当前数据集的类别统计信息
            current_stats = dataset.get_dataset_stats()
            category_stats = current_stats.get('category_stats', {})
            
            # 提取图片名称和结果
            img_names = list(unlabeled_results.keys())
            results = [info['result'] for info in unlabeled_results.values()]
            
            # 获取配置的样本数量，而不是使用sampling_budget
            num_samples = cfg.active_learning.sample_selection.get('num_samples', 200)
            
            # 使用MUS-CDB的rank_samples_with_cdb方法，但限制总预算为num_samples
            selected_samples = selector.rank_samples_with_cdb(
                results=results,
                labeled_counts=category_stats,
                img_names=img_names,
                total_budget=num_samples,  # 使用num_samples作为预算限制
                theta=cfg.active_learning.inference_options.get('score_thr', 0.1)
            )
            
            # 确保不超过num_samples
            if len(selected_samples) > num_samples:
                selected_samples = selected_samples[:num_samples]
                logger.info(f"MUS-CDB策略选择了{len(selected_samples)}个样本，已限制到{num_samples}个")
        else:
            # 传统单阶段策略
            selected_samples = selector.select_samples(
                unlabeled_results=unlabeled_results,
                num_samples=cfg.active_learning.sample_selection['num_samples']
            )
        
        # 生成并保存选择报告
        report = selector.get_selection_report(selected_samples, unlabeled_results)
        logger.debug(f"选择报告:\n{report}")

        try:
            ranking_dir = Path(cfg.work_dir) / 'ranking_results'
            ranking_dir.mkdir(parents=True, exist_ok=True)
            uncertainty_metric = cfg.active_learning.sample_selection.get('uncertainty_metric', 'default')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ranking_file = ranking_dir / f'ranking_{uncertainty_metric}_{timestamp}.json'
            
            # 保存JSON文件
            with open(ranking_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.debug(f"排序结果已保存到: {ranking_file}")
        except Exception as e:
            logger.error(f"保存排序结果失败: {str(e)}")
            
        return selected_samples
        
    # 处理其他选择器类型
    unlabeled_stats = convert_unlabeled_to_stats_format(unlabeled_results)
    
    if selector_type == 'Wasserstein':
        return selector.select_samples(
            unlabeled_stats, 
            select_num=cfg.active_learning.sample_selection['num_samples']
        )
        
    elif selector_type == 'RL':
        q_table_path = str(Path(cfg.work_dir) / 'rl_q_table.json')
        return selector.select_samples(
            unlabeled_stats, 
            select_num=cfg.active_learning.sample_selection['num_samples'],
            q_table_path=q_table_path
        )
        
    elif selector_type == 'Combinatorial':
        selected_samples, report = selector.select_samples(
            train_stats=train_stats,
            unlabeled_stats=unlabeled_stats,
            select_num=cfg.active_learning.sample_selection['num_samples']
        )
        log_combinatorial_report(report, logger)
        return selected_samples
        
    raise ValueError(f"未知的选择器类型: {selector_type}")

def log_combinatorial_report(report: Dict, logger: MMLogger):
    """记录组合选择器的报告"""
    logger.info(f"{'='*20} Combinatorial Selection Report {'='*20}")
    logger.info(f"Selected samples count: {report['selected_count']}")
    logger.info(f"Average SSC score: {report['avg_ssc']:.4f}")
    logger.info("Feature Changes:")
    logger.info(f"{'Feature':<20} {'Change':<15} {'Change %':<15}")
    logger.info("-" * 50)
    
    for metric, data in report['feature_changes'].items():
        logger.info(
            f"{metric:<20} {data['change']:>+.4f}      {data['change_pct']:>+.2f}%"
        )
    logger.info("=" * 65)
    
    if 'optimization_status' in report:
        logger.info(f"Optimization status: {report['optimization_status']}")

def fallback_selection(dataset: ActiveCocoDataset, unlabeled_results: Dict, cfg: Config, logger: MMLogger) -> List[str]:
    """备用样本选择方法"""
    logger.warning("使用基础默认方法作为备选")
    selected_samples = dataset.select_samples(
        results=unlabeled_results,
        num_samples=cfg.active_learning.sample_selection['num_samples'],
        uncertainty_metric=cfg.active_learning.sample_selection['uncertainty_metric']
    )
    if not selected_samples:
        raise RuntimeError("备选方法也未能选择样本")
    logger.info(f"备选方法完成 - 选择了 {len(selected_samples)} 个样本")
    return selected_samples

def select_samples(train_stats: Dict, unlabeled_results: Dict, cfg: Config, logger: MMLogger) -> List[str]:
    """选择样本"""
    al_cfg = cfg.active_learning
    dataset = ActiveCocoDataset(
        data_root=al_cfg.data_root,
        ann_file=cfg.train_dataloader.dataset.ann_file,
        data_prefix=cfg.train_dataloader.dataset.data_prefix
    )
    
    selector_type = al_cfg.sample_selection['sample_selector']
    selector = create_selector(selector_type, train_stats, al_cfg, logger)
    
    try:
        # 传入train_stats参数
        selected_samples = run_selector(selector, dataset, unlabeled_results, train_stats, cfg, logger)
        if not selected_samples:
            raise ValueError("未选择到样本")
        return selected_samples
    except Exception as e:
        logger.error(f"样本选择过程错误: {str(e)}")
        return fallback_selection(dataset, unlabeled_results, cfg, logger)

def update_dataset(selected_samples: List[str], cfg: Config, logger: MMLogger) -> Dict:
    """更新数据集"""
    al_cfg = cfg.active_learning
    dataset = ActiveCocoDataset(
        data_root=al_cfg.data_root,
        ann_file=cfg.train_dataloader.dataset.ann_file,
        data_prefix=cfg.train_dataloader.dataset.data_prefix
    )
    
    # 更新数据集
    success = dataset.update_dataset(selected_samples)
    if not success:
        raise RuntimeError("数据集更新失败")
    logger.info("数据集更新成功")
    
    # 获取当前统计信息
    current_stats = dataset.get_dataset_stats()
    return current_stats

def update_performance_history(performance_history: Dict, current_stats: Dict, eval_results: Dict, round_num: int) -> None:
    """更新性能历史"""
    # 更新性能历史
    performance_history['round'].append(round_num)
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


def save_round_stats(
    work_dir: Path, 
    round_num: int, 
    selected_samples: List[str], 
    eval_results: Dict, 
    performance_history: Dict, 
    logger: MMLogger,
    cfg: Config  # 新增参数：直接传入配置对象
) -> None:
    """保存轮次统计信息"""
    # 创建轮次工作目录
    iter_work_dir = work_dir / f"round_{round_num}"
    iter_work_dir.mkdir(exist_ok=True)
    
    # 获取当前数据集统计 - 直接使用传入的配置对象
    al_cfg = cfg.active_learning
    # 只有data_root是实际参与处理并被脚本使用的核心参数，
    # 而ann_file和data_prefix主要用于保证类与框架的兼容性，
    # 在主动学习过程中不直接参与数据管理。
    dataset = ActiveCocoDataset(
        data_root=al_cfg.data_root,
        ann_file=cfg.train_dataloader.dataset.ann_file,  # 使用配置中的路径
        data_prefix=cfg.train_dataloader.dataset.data_prefix  # 使用配置中的路径
    )
    current_stats = dataset.get_dataset_stats()
    
    # 保存统计信息
    stats_info = {
        'iteration': round_num,
        'selected_samples': selected_samples,
        'dataset_stats': current_stats,
        'evaluation_results': eval_results
    }
    
    with open(iter_work_dir / 'stats.json', 'w') as f:
        json.dump(stats_info, f, indent=2)
    
    # 保存性能历史到CSV
    df = pd.DataFrame(performance_history)
    df.to_csv(work_dir / 'performance_history.csv', index=False)
    
    # 打印当前轮次的详细信息
    print_round_summary(current_stats, eval_results, logger)

def print_round_summary(current_stats: Dict, eval_results: Dict, logger: MMLogger):
    """打印轮次总结信息"""
    val_results = eval_results.get('val', {})
    
    logger.info("\n当前轮次统计信息:")
    logger.info("数据集统计:")
    logger.info(f"  - 已标注图片数: {current_stats['labeled_images']}")
    logger.info(f"  - 未标注图片数: {current_stats['unlabeled_images']}")
    logger.info(f"  - 总图片数: {current_stats['total_images']}")
    logger.info(f"  - 标注比例: {current_stats['labeled_ratio']:.2%}")
    logger.info(f"  - 已标注框数量: {current_stats['labeled_annotations']}")
    
    if val_results:
        logger.info("验证集性能:")
        logger.info(f"  - bbox_mAP: {val_results.get('bbox_mAP', 0.0):.4f}")
        logger.info(f"  - bbox_mAP_50: {val_results.get('bbox_mAP_50', 0.0):.4f}")
        logger.info(f"  - bbox_mAP_75: {val_results.get('bbox_mAP_75', 0.0):.4f}")

def create_checkpoint_file(iter_work_dir: Path):
    """创建检查点文件，用于标记当前轮次的进度"""
    checkpoint_states = {
        'started': True,
        'training_done': False,
        'inference_done': False,
        'selection_done': False,
        'completed': False,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        with open(iter_work_dir / 'checkpoint_state.json', 'w') as f:
            json.dump(checkpoint_states, f, indent=2)
    except Exception as e:
        print(f"保存检查点状态文件失败: {e}")

def update_checkpoint_state(iter_work_dir: Path, state_key: str):
    """更新检查点状态"""
    checkpoint_file = iter_work_dir / 'checkpoint_state.json'
    if not checkpoint_file.exists():
        create_checkpoint_file(iter_work_dir)
        
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint_states = json.load(f)
            
        checkpoint_states[state_key] = True
        checkpoint_states['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_states, f, indent=2)
    except Exception as e:
        print(f"更新检查点状态失败: {e}")