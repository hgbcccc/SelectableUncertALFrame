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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='主动学习训练')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--work-dir', help='工作目录')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='覆盖配置文件中的选项')
    parser.add_argument('--resume', action='store_true', help='是否从中断处恢复训练')
    args = parser.parse_args()
    return args


class DataHandler:
    """数据处理类"""

    def __init__(self, al_cfg, cfg):
        self.al_cfg = al_cfg
        self.cfg = cfg

    def update_dataset(self, dataset: ActiveCocoDataset, selected_samples: List[str]) -> bool:
        return dataset.update_dataset(selected_samples)

    def get_dataset_stats(self, dataset: ActiveCocoDataset) -> Dict:
        return dataset.get_dataset_stats()


class ModelHandler:
    """模型处理类"""

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

    def _load_previous_checkpoint(self, round_num: int, work_dir: Path):
        prev_iter_dir = work_dir / f"round_{round_num - 1}"
        prev_ckpt = find_best_checkpoint(prev_iter_dir, self.logger)
        if prev_ckpt:
            self.logger.info(f"加载上一轮检查点: {prev_ckpt}")
            self.cfg.load_from = prev_ckpt
        else:
            self.logger.warning("未找到上一轮检查点")

    def train_model(self, iter_work_dir: Path, round_num: int) -> Tuple[Runner, Dict]:
        self.cfg.score_threshold, self.cfg.max_boxes_per_img, self.cfg.nms_iou_threshold = \
            adjust_thresholds(self.cfg, round_num - 1, self.cfg.active_learning.max_iterations)

        if round_num > 1:
            self._load_previous_checkpoint(round_num, iter_work_dir)

        runner = Runner.from_cfg(self.cfg)
        runner.train()

        eval_results = self.evaluate_model(runner)

        return runner, eval_results

    def evaluate_model(self, runner: Runner) -> Dict:
        eval_results = {'val': {}}
        try:
            if hasattr(self.cfg, 'val_dataloader') and hasattr(self.cfg, 'val_evaluator'):
                val_results = runner.val()
                if isinstance(val_results, dict):
                    eval_results['val'] = {
                        'bbox_mAP': val_results.get('coco/bbox_mAP', 0.0),
                        'bbox_mAP_50': val_results.get('coco/bbox_mAP_50', 0.0),
                        'bbox_mAP_75': val_results.get('coco/bbox_mAP_75', 0.0),
                        'bbox_mAP_95': val_results.get('coco/bbox_mAP_95', 0.0)
                    }
                formatted_result = ", ".join([f"{key}: {value}" for key, value in eval_results['val'].items()])
                self.logger.info(f"验证集评估结果: {formatted_result}")
        except Exception as e:
            self.logger.warning(f"评估过程出错: {e}")
        return eval_results


class UncertaintyHandler:
    """不确定性处理类"""

    def __init__(self, al_cfg, args, logger, cfg):
        self.al_cfg = al_cfg
        self.args = args
        self.logger = logger
        self.cfg = cfg  # 添加这一行，保存配置对象
        self.round = 1

    def _calculate_train_stats(self, iter_work_dir: Path) -> Dict:
        self.logger.info("开始计算训练数据SSC分数...")
        calculator = SSCCalculator(data_cfg=self.al_cfg.train_pool_cfg)
        train_stats = calculator.calculate_dataset_stats()
        train_stats = normalize_train_stats(train_stats)

        stats_path = iter_work_dir / 'train_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(train_stats, f, indent=2, ensure_ascii=False)

        return train_stats

    def _infer_unlabeled_data(self, iter_work_dir: Path) -> Dict:
        self.logger.info("开始推理未标注数据...")
        latest_ckpt = find_best_checkpoint(iter_work_dir, self.logger)
        if not latest_ckpt:
            raise FileNotFoundError(f"在 {iter_work_dir} 中未找到有效的检查点文件")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        teacher_output_dir = Path(self.cfg.work_dir) / f'round_{self.round}' / 'teacher_outputs'
        teacher_output_dir.mkdir(parents=True, exist_ok=True)

        teacher = SimDetectionInference(
            config_file=self.args.config,
            batch_size=self.al_cfg.inference_options.batch_size,
            checkpoint_file=latest_ckpt,
            output_dir=str(teacher_output_dir),
            enable_uncertainty=True,
            uncertainty_methods=self.al_cfg.inference_options.uncertainty_methods
        )

        results = teacher.inference(
            str(Path(self.al_cfg.data_root) / 'images_unlabeled'),
            sample_size=self.al_cfg.inference_options.sample_size
        )

        # uncertainty_calculator = UncertaintyMetrics()
        # uncertainty_methods = self.al_cfg.inference_options.get('uncertainty_methods', ['all'])
        # score_thr = self.al_cfg.inference_options.get('score_thr', 0.09)

        # processed_results = {}
        # for img_name, result_info in tqdm(results.items(), desc="计算不确定性"):
        #     try:
        #         result = result_info['result']
        #         uncertainty = uncertainty_calculator.compute_uncertainty(
        #             result,
        #             methods=uncertainty_methods,
        #             min_score_thresh=score_thr
        #         )
        #         processed_results[img_name] = {
        #             'result': result,
        #             'vis_path': result_info.get('vis_path'),
        #             'uncertainty': uncertainty
        #         }
        #     except Exception as e:
        #         self.logger.error(f"计算不确定性时出错 ({img_name}): {str(e)}")
        #         processed_results[img_name] = result_info

        # uncertainty_dir = teacher_output_dir / 'uncertainty'
        # uncertainty_dir.mkdir(parents=True, exist_ok=True)
        # uncertainty_path = uncertainty_dir / 'uncertainty_results.json'

        # with open(uncertainty_path, 'w', encoding='utf-8') as f:
        #     json.dump({
        #         'timestamp': timestamp,
        #         'uncertainty_methods': uncertainty_methods,
        #         'results': {
        #             img_name: {
        #                 'uncertainty': info['uncertainty']
        #             } for img_name, info in processed_results.items()
        #             if 'uncertainty' in info
        #         }
        #     }, f, indent=2, ensure_ascii=False)

        # self.logger.info(f"不确定性结果已保存至: {uncertainty_path}")
        # results = processed_results

        # results = recalculate_crown_count_scores(results, results)
        # results = normalize_uncertainty_scores(results)

        # return results
        # 替换原有的不确定性计算部分

        uncertainty_methods = self.al_cfg.inference_options.uncertainty_methods
        score_thr = self.al_cfg.inference_options.get('score_thr', 0.09)

        if 'mus' in uncertainty_methods or 'all' in uncertainty_methods:
            # 使用两阶段MUS-CDB采样策略
            uncertainty_calculator = MUSCDBSampling()
            use_two_stage = True
            
            # 获取当前数据集统计（不保存到文件，直接使用内存数据）
            dataset = ActiveCocoDataset(
                data_root=self.al_cfg.data_root,
                ann_file=self.cfg.train_dataloader.dataset.ann_file,
                data_prefix=self.cfg.train_dataloader.dataset.data_prefix
            )
            data_handler = DataHandler(self.al_cfg, self.cfg)
            current_stats = data_handler.get_dataset_stats(dataset)
            
        else:
            # 使用原有的单阶段采样策略
            uncertainty_calculator = UncertaintyMetrics()
            use_two_stage = False

        # 计算不确定性（保持原有逻辑）
        processed_results = {}
        for img_name, result_info in tqdm(results.items(), desc="计算不确定性"):
            try:
                result = result_info['result']
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
                self.logger.error(f"计算不确定性时出错 ({img_name}): {str(e)}")
                processed_results[img_name] = result_info

        # 保存不确定性结果（保持原有逻辑）
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

        self.logger.info(f"不确定性结果已保存至: {uncertainty_path}")
        results = processed_results

        # 后处理（保持原有逻辑）
        results = recalculate_crown_count_scores(results, results)
        results = normalize_uncertainty_scores(results)

        # 使用MUS-CDB策略选择样本（仅在启用两阶段策略时执行）
        if use_two_stage:
            # 修改MUSCDBSampling类，支持直接传入统计数据
            unlabeled_results = [info['result'] for info in results.values()]
            selected_bbox_ids = uncertainty_calculator.rank_samples_with_cdb(
                results=unlabeled_results,
                labeled_counts=current_stats["category_stats"],  # 直接传入类别统计
                total_budget=self.al_cfg.sample_selection.sampling_budget,
                theta=0.1
            )
            
            # 更新选中样本信息
            for img_name, info in results.items():
                info['selected'] = [bbox_id for bbox_id in info['result'].pred_instances if bbox_id in selected_bbox_ids]

        return results

    def process_unlabeled_data(self, iter_work_dir: Path) -> Tuple[Dict, Dict]:
        train_stats = {}
        selector_type = self.al_cfg.sample_selection['sample_selector']
        if self._check_need_ssc():
            self.logger.info(f"计算训练集统计信息 (由{selector_type}选择器或SSC度量要求)")
            train_stats = self._calculate_train_stats(iter_work_dir)
        else:
            self.logger.debug(f"使用{selector_type}选择器，不需要训练集统计信息")

        unlabeled_results = self._infer_unlabeled_data(iter_work_dir)
        unlabeled_stats = convert_unlabeled_to_stats_format(unlabeled_results)
        unlabeled_stats = normalize_train_stats(unlabeled_stats)

        for feat in ['ssc_score', 'occlusion_score', 'crown_count_score', 'diversity_score', 'area_var_score',
                     'density_var_score']:
            mean = unlabeled_stats['features'][feat]['mean']
            distribution = unlabeled_stats['features'][feat]['distribution']
            is_unique = len(set(distribution)) == 1
            self.logger.info(f"{feat} mean: {mean}")
            self.logger.info(f"{feat} is unique: {is_unique}")

        unlabeled_stats_path = iter_work_dir / 'unlabeled_stats.json'
        try:
            with open(unlabeled_stats_path, 'w', encoding='utf-8') as f:
                json.dump(unlabeled_stats, f, indent=2, ensure_ascii=False)
            self.logger.info(f"未标注数据统计信息已保存至: {unlabeled_stats_path}")
        except Exception as e:
            self.logger.error(f"保存未标注数据统计信息失败: {str(e)}")

        return train_stats, unlabeled_results

    def _check_need_ssc(self) -> bool:
        selector_type = self.al_cfg.sample_selection.get('sample_selector', '')
        selected_metric = self.al_cfg.inference_options.get('selected_metric', '')

        need_train_stats_selectors = {
            'Combinatorial',
            'Wasserstein',
            'RL'
        }

        return (
                selector_type in need_train_stats_selectors or
                selected_metric =='ssc_score'
        )


class SampleSelector:
    """样本选择类"""

    def __init__(self, al_cfg, logger, cfg):
        self.al_cfg = al_cfg
        self.logger = logger
        self.cfg = cfg  # 添加这一行，保存配置对象

    def _create_selector(self, selector_type: str, train_stats: Dict):
        if selector_type == 'default':
            self.logger.info(f"使用默认选择器")
            return DefaultSelector(self.al_cfg.sample_selection['uncertainty_metric'])
        elif selector_type == 'Wasserstein':
            self.logger.info(f"使用Wasserstein选择器")
            return WassersteinSampleSelector(train_stats)
        elif selector_type == 'RL':
            self.logger.info(f"使用RL选择器")
            return RLSampleSelector(
                train_stats,
                metric=self.al_cfg.sample_selection.get('rl_metric', 'wasserstein')
            )
        elif selector_type == 'Combinatorial':
            self.logger.info(f"使用组合选择器")
            combo_cfg = self.al_cfg.sample_selection.get('combinatorial', {})
            return CombinatorialSelector(
                feature_weights=combo_cfg.get('feature_weights'),
                ssc_weight=combo_cfg.get('ssc_weight', 0.7)
            )
        else:
            raise ValueError(f"未知的选择器类型: {selector_type}")

    def _run_selector(self, selector, dataset, unlabeled_results, train_stats) -> List[str]:
        selector_type = self.al_cfg.sample_selection['sample_selector']

        if selector_type == 'default':
            selected_samples = selector.select_samples(
                unlabeled_results=unlabeled_results,
                num_samples=self.al_cfg.sample_selection['num_samples']
            )
            report = selector.get_selection_report(selected_samples, unlabeled_results)
            self.logger.debug(f"report:\n{report}")

            try:
                ranking_dir = Path(self.cfg.work_dir) / 'ranking_results'
                ranking_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                uncertainty_metric = self.al_cfg.sample_selection.get('uncertainty_metric', 'default')
                ranking_file = ranking_dir / f'ranking_{uncertainty_metric}_{timestamp}.json'

                with open(ranking_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                self.logger.debug(f"排序结果已保存到: {ranking_file}")

            except Exception as e:
                self.logger.error(f"保存排序结果失败: {str(e)}")
            return selected_samples

        unlabeled_stats = convert_unlabeled_to_stats_format(unlabeled_results)

        if selector_type == 'Wasserstein':
            return selector.select_samples(
                unlabeled_stats,
                select_num=self.al_cfg.sample_selection['num_samples']
            )
        elif selector_type == 'RL':
            q_table_path = str(self.work_dir / 'rl_q_table.json')
            return selector.select_samples(
                unlabeled_stats,
                select_num=self.al_cfg.sample_selection['num_samples'],
                q_table_path=q_table_path
            )
        elif selector_type == 'Combinatorial':
            selected_samples, report = selector.select_samples(
                train_stats=train_stats,
                unlabeled_stats=unlabeled_stats,
                select_num=self.al_cfg.sample_selection['num_samples']
            )
            self._log_combinatorial_report(report)
            return selected_samples

        raise ValueError(f"未知的选择器类型: {selector_type}")

    def _log_combinatorial_report(self, report: Dict):
        self.logger.info(f"{'=' * 20} Combinatorial Selection Report {'=' * 20}")
        self.logger.info(f"Selected samples count: {report['selected_count']}")
        self.logger.info(f"Average SSC score: {report['avg_ssc']:.4f}")
        self.logger.info("Feature Changes:")
        self.logger.info(f"{'Feature':<20} {'Change':<15} {'Change %':<15}")
        self.logger.info("-" * 50)

        for metric, data in report['feature_changes'].items():
            self.logger.info(
                f"{metric:<20} {data['change']:>+.4f}      {data['change_pct']:>+.2f}%"
            )
        self.logger.info("=" * 65)

        if 'optimization_status' in report:
            self.logger.info(f"Optimization status: {report['optimization_status']}")

    def _fallback_selection(self, dataset: ActiveCocoDataset, unlabeled_results: Dict) -> List[str]:
        self.logger.warning("使用基础默认方法作为备选")
        selected_samples = dataset.select_samples(
            results=unlabeled_results,
            num_samples=self.al_cfg.sample_selection['num_samples'],
            uncertainty_metric=self.al_cfg.sample_selection['uncertainty_metric']
        )
        if not selected_samples:
            raise RuntimeError("备选方法也未能选择样本")
        self.logger.info(f"备选方法完成 - 选择了 {len(selected_samples)} 个样本")
        return selected_samples

    def select_samples(self, train_stats: Dict, unlabeled_results: Dict) -> List[str]:
        dataset = ActiveCocoDataset(
            data_root=self.al_cfg.data_root,
            ann_file=self.cfg.train_dataloader.dataset.ann_file,
            data_prefix=self.cfg.train_dataloader.dataset.data_prefix
        )

        selector_type = self.al_cfg.sample_selection['sample_selector']
        selector = self._create_selector(selector_type, train_stats)

        try:
            selected_samples = self._run_selector(selector, dataset, unlabeled_results, train_stats)
            if not selected_samples:
                raise ValueError("未选择到样本")
            return selected_samples
        except Exception as e:
            self.logger.error(f"样本选择过程错误: {str(e)}")
            return self._fallback_selection(dataset, unlabeled_results)


class ActiveLearningTrainer:
    """主动学习训练器"""

    def __init__(self, cfg: Config, args: argparse.Namespace):
        self.cfg = cfg
        self.args = args
        self.logger = MMLogger.get_current_instance()
        self.work_dir = self._setup_work_dir()
        self.al_cfg = cfg.active_learning
        self.performance_history = self._init_performance_history()
        self.need_ssc = UncertaintyHandler(self.al_cfg, self.args, self.logger, self.cfg)._check_need_ssc()
        # 初始化轮次计数器
        self.round = 1
        self.max_iterations = self.al_cfg.get('max_iterations', 10)

        # 检查恢复状态
        if args.resume:
            self._check_resume_state()

    def _check_resume_state(self):
        """检查训练状态，决定从哪一轮恢复训练"""
        self.logger.info("检查训练恢复状态...")

        # 查找现有轮次目录
        round_dirs = sorted([d for d in self.work_dir.glob("round_*") if d.is_dir()],
                            key=lambda x: int(str(x).split("_")[-1]))

        if not round_dirs:
            self.logger.info("未找到现有训练轮次，将从第1轮开始训练")
            return

        # 检查最后一轮是否完成
        last_round_dir = round_dirs[-1]
        last_round_num = int(str(last_round_dir).split("_")[-1])

        # 检查是否存在stats.json文件，该文件表示轮次已完成
        stats_file = last_round_dir /'stats.json'
        if stats_file.exists():
            # 最后一轮已完成，从下一轮开始
            self.round = last_round_num + 1
            self.logger.info(f"检测到第{last_round_num}轮已完成，将从第{self.round}轮开始训练")
        else:
            # 最后一轮未完成，从该轮重新开始
            self.round = last_round_num
            self.logger.info(f"检测到第{last_round_num}轮未完成，将从第{self.round}轮重新开始训练")

        # 加载性能历史记录
        history_csv = self.work_dir / 'performance_history.csv'
        if history_csv.exists():
            try:
                df = pd.read_csv(history_csv)
                for col in self.performance_history.keys():
                    if col in df.columns:
                        self.performance_history[col] = df[col].tolist()
                self.logger.info(f"已加载现有性能历史记录，包含{len(df)}轮数据")
            except Exception as e:
                self.logger.warning(f"加载性能历史记录失败: {e}")

        # 如果超过最大轮次，结束训练
        if self.round > self.max_iterations:
            self.logger.info(f"已完成所有{self.max_iterations}轮训练，无需继续")
            exit(0)

    def _setup_work_dir(self) -> Path:
        """设置工作目录"""
        if self.args.work_dir is not None:
            self.cfg.work_dir = self.args.work_dir
        elif self.cfg.get('work_dir', None) is None:
            self.cfg.work_dir = Path('./work_dirs') / Path(self.args.config).stem

        work_dir = Path(self.cfg.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def _init_performance_history(self) -> Dict:
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

    def train(self):
        data_handler = DataHandler(self.al_cfg, self.cfg)
        model_handler = ModelHandler(self.cfg, self.logger)
        uncertainty_handler = UncertaintyHandler(self.al_cfg, self.args, self.logger, self.cfg)
        sample_selector = SampleSelector(self.al_cfg, self.logger, self.cfg)

        for self.round in range(self.round, self.max_iterations + 1):
            self.logger.info(f"\n开始第 {self.round}/{self.max_iterations} 轮主动学习...")

            # 创建轮次工作目录
            iter_work_dir = self.work_dir / f"round_{self.round}"
            iter_work_dir.mkdir(exist_ok=True)
            self.cfg.work_dir = str(iter_work_dir)

            # 训练和评估
            self.logger.info(f"第 {self.round} 轮训练开始...")
            _, eval_results = model_handler.train_model(iter_work_dir, self.round)

            # 处理未标注数据
            self.logger.info(f"第 {self.round} 轮推理开始...")
            train_stats, unlabeled_results = uncertainty_handler.process_unlabeled_data(iter_work_dir)

            # 选择样本
            self.logger.info(f"第 {self.round} 轮样本选择开始...")
            selected_samples = sample_selector.select_samples(train_stats, unlabeled_results)

            # 更新数据集和性能历史
            self.logger.info(f"第 {self.round} 轮更新数据集...")
            dataset = ActiveCocoDataset(
                data_root=self.al_cfg.data_root,
                ann_file=self.cfg.train_dataloader.dataset.ann_file,
                data_prefix=self.cfg.train_dataloader.dataset.data_prefix
            )
            success = data_handler.update_dataset(dataset, selected_samples)
            if not success:
                raise RuntimeError("数据集更新失败")
            self.logger.info("数据集更新成功")

            current_stats = data_handler.get_dataset_stats(dataset)

            self.performance_history['round'].append(self.round)
            self.performance_history['labeled_ratio'].append(current_stats['labeled_ratio'])
            self.performance_history['labeled_images'].append(current_stats['labeled_images'])
            self.performance_history['unlabeled_images'].append(current_stats['unlabeled_images'])
            self.performance_history['total_images'].append(current_stats['total_images'])
            self.performance_history['labeled_annotations'].append(current_stats['labeled_annotations'])

            val_results = eval_results.get('val', {})
            self.performance_history['val_bbox_mAP'].append(val_results.get('bbox_mAP', 0.0))
            self.performance_history['val_bbox_mAP_50'].append(val_results.get('bbox_mAP_50', 0.0))
            self.performance_history['val_bbox_mAP_75'].append(val_results.get('bbox_mAP_75', 0.0))

            self.performance_history['timestamp'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            # 保存本轮统计信息
            self._save_round_stats(iter_work_dir, self.round, selected_samples, eval_results)

            # 清理GPU内存
            torch.cuda.empty_cache()

            self.logger.info(f"第 {self.round} 轮已完成")

    def _save_round_stats(self, iter_work_dir: Path, round_num: int, selected_samples: List[str], eval_results: Dict):
        """保存轮次统计信息"""
        # 获取当前数据集统计
        dataset = ActiveCocoDataset(
            data_root=self.al_cfg.data_root,
            ann_file=self.cfg.train_dataloader.dataset.ann_file,
            data_prefix=self.cfg.train_dataloader.dataset.data_prefix
        )
        data_handler = DataHandler(self.al_cfg, self.cfg)
        current_stats = data_handler.get_dataset_stats(dataset)

        # 保存统计信息
        stats_info = {
            'iteration': round_num,
            'selected_samples': selected_samples,
            'dataset_stats': current_stats,
            'evaluation_results': eval_results
        }

        with open(iter_work_dir /'stats.json', 'w') as f:
            json.dump(stats_info, f, indent=2)

        # 保存性能历史到CSV
        df = pd.DataFrame(self.performance_history)
        df.to_csv(self.work_dir / 'performance_history.csv', index=False)

        # 打印当前轮次的详细信息
        self._print_round_summary(current_stats, eval_results)

    def _print_round_summary(self, current_stats: Dict, eval_results: Dict):
        """打印轮次总结信息"""
        val_results = eval_results.get('val', {})

        self.logger.info("\n当前轮次统计信息:")
        self.logger.info("数据集统计:")
        self.logger.info(f"  - 已标注图片数: {current_stats['labeled_images']}")
        self.logger.info(f"  - 未标注图片数: {current_stats['unlabeled_images']}")
        self.logger.info(f"  - 总图片数: {current_stats['total_images']}")
        self.logger.info(f"  - 标注比例: {current_stats['labeled_ratio']:.2%}")
        self.logger.info(f"  - 已标注框数量: {current_stats['labeled_annotations']}")

        if val_results:
            self.logger.info("验证集性能:")
            self.logger.info(f"  - bbox_mAP: {val_results.get('bbox_mAP', 0.0):.4f}")
            self.logger.info(f"  - bbox_mAP_50: {val_results.get('bbox_mAP_50', 0.0):.4f}")
            self.logger.info(f"  - bbox_mAP_75: {val_results.get('bbox_mAP_75', 0.0):.4f}")


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    trainer = ActiveLearningTrainer(cfg, args)
    trainer.train()


if __name__ == '__main__':
    import sys

    # sys.argv = ['sual/active_learnning_train_loop.py', 'custom_config/faster - rcnn_margin_default.py',
    #             '--work-dir', 'work_dirs/faster - rcnn_margin_default']
    main()        


    # python sual/main.py custom_config/cascade-rcnn_mus_cdb.py --work-dir cascade-rcnn_mus_cdb_16_10                                      