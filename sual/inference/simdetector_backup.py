import warnings
import pickle
warnings.filterwarnings("ignore")
import argparse
import os
import random 
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Union, List, Dict, Optional
import mmcv
import torch
import numpy as np
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample
from sual.core.uncertainty.metrics import UncertaintyMetrics


class SimDetectionInference:
    """简化版目标检测推理器"""
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def __init__(
            self,
            config_file: str,
            checkpoint_file: str,
            device: str = 'cuda:0',
            output_dir: str = 'outputs',
            batch_size: int = 4,
            num_workers: int = 4,
            score_thr: float = 0.2,
            enable_uncertainty: bool = True,  # 添加不确定性参数
            enable_mc_dropout: bool = False,
            mc_dropout_times: int = 20,
            uncertainty_methods: Optional[List[str]] = None,
            sample_size: int = 0,
    ):
        # 注册模块
        register_all_modules()
        
        # 基础配置
        self.device = device
        self.batch_size = batch_size
        self.score_thr = score_thr
        
        # 初始化模型
        self.model = init_detector(config_file, checkpoint_file, device=device)
        
        # 设置输出目录
        self._setup_directories(output_dir)

        # 设置日志
        self.logger = self._setup_logging()
        
        # MC Dropout 配置
        self.enable_mc_dropout = enable_mc_dropout
        self.mc_dropout_times = mc_dropout_times
        
        # 初始化可视化器
        self._setup_visualizer()
        self.sample_size = sample_size
        


        # 不确定性分析配置
        self.enable_uncertainty = enable_uncertainty
        if enable_uncertainty:
            self.uncertainty_metrics = UncertaintyMetrics()
            self.uncertainty_methods = uncertainty_methods or ['all']

    def _setup_directories(self, output_dir: str):
        """设置输出目录结构"""
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_output_dir = self.output_dir / self.timestamp
        self.current_output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        self.results_dir = self.current_output_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
        self.vis_dir = self.current_output_dir / 'visualize'
        self.vis_dir.mkdir(exist_ok=True)
        self.uncertainty_dir = self.current_output_dir / 'uncertainty'
        self.uncertainty_dir.mkdir(exist_ok=True)

    def _setup_logging(self):
        """配置日志系统"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        fh = logging.FileHandler(self.current_output_dir / 'inference.log')
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def _setup_visualizer(self):
        """配置可视化器"""
        cfg = self.model.cfg.copy()
        cfg.visualizer.save_dir = str(self.vis_dir)
        self.visualizer = VISUALIZERS.build(cfg.visualizer)
        self.visualizer.dataset_meta = self.model.dataset_meta

    def _get_image_files(self, input_path: Union[str, Path]) -> List[Path]:
        """获取所有支持的图片文件路径"""
        input_path = Path(input_path)
        if input_path.is_file():
            if input_path.suffix.lower() in self.SUPPORTED_FORMATS:
                return [input_path]
            raise ValueError(f"不支持的文件格式: {input_path}")
        
        image_files = []
        for fmt in self.SUPPORTED_FORMATS:
            image_files.extend(input_path.glob(f"*{fmt.lower()}"))
        return sorted(image_files)

    def _visualize_result(self, img: np.ndarray, result: DetDataSample, 
                         output_path: str) -> Optional[str]:
        """可视化结果"""
        try:
            visualizer = VISUALIZERS.build(self.model.cfg.visualizer)
            visualizer.dataset_meta = self.model.dataset_meta

            # 限制图片大小
            h, w = img.shape[:2]
            max_size = 2048
            if max(h, w) > max_size:
                ratio = max_size / max(h, w)
                img = mmcv.imresize(img, (int(w * ratio), int(h * ratio)))

            visualizer.add_datasample(
                name='',
                image=img,
                data_sample=result,
                draw_gt=False,
                pred_score_thr=self.score_thr,
                show=False,
                wait_time=0,
                out_file=None
            )

            vis_img = visualizer.get_image()
            mmcv.imwrite(vis_img, output_path)
            return output_path
        except Exception as e:
            self.logger.error(f"可视化失败: {e}")
            return None

    def inference(
            self,
            input_path: Union[str, Path],
            save_results: bool = True,
            save_vis: bool = True,
            score_thr:int = 0.09,
            batch_size :int = 4,
            sample_size: int = 0,
    ) -> Dict[str, Dict]:
        """执行推理"""
        # 获取图片文件
        image_files = self._get_image_files(input_path)
        # self.logger.info(f"找到 {len(image_files)} 张图片")
        # print(f"找到 {len(image_files)} 张图片")
        if not image_files:
            raise ValueError(f"在 {input_path} 中没有找到支持的图片文件")

        # 处理采样逻辑
        if sample_size > 0 and sample_size < len(image_files):
            # self.logger.info(f"采样 {sample_size} 张图片进行推理 (总共 {len(image_files)} 张)")
            # print(f"采样 {sample_size} 张图片进行推理 (总共 {len(image_files)} 张)")
            image_files = random.sample(image_files, sample_size)

        results = {}
        pbar = tqdm(total=len(image_files), desc="处理图片")
        
        # 批量处理图片
        for i in range(0, len(image_files), self.batch_size):
            batch_paths = image_files[i:i + self.batch_size]
            
            for img_path in batch_paths:
                try:
                    # 读取图片
                    img = mmcv.imread(str(img_path))
                    if img is None:
                        self.logger.warning(f"无法读取图片: {img_path}")
                        continue
                    
                    # 执行推理
                    self.model.eval()
                    with torch.no_grad():
                        result = inference_detector(self.model, img)
                    
                    result_info = {
                        'result': result,
                        'vis_path': None
                    }
                    
                    # 可视化
                    if save_vis:
                        vis_path = str(self.vis_dir / f"{img_path.stem}_vis.jpg")
                        vis_path = self._visualize_result(img, result, vis_path)
                        result_info['vis_path'] = vis_path
                    
                    # 保存结果
                    if save_results:
                        json_path = self.results_dir / f"{img_path.stem}_result.json"
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump({
                                'scores': result.pred_instances.scores.cpu().numpy().tolist(),
                                'labels': result.pred_instances.labels.cpu().numpy().tolist(),
                                'bboxes': result.pred_instances.bboxes.cpu().numpy().tolist(),
                                'labels_names': [
                                    self.model.dataset_meta['classes'][label]
                                    for label in result.pred_instances.labels.cpu().numpy()
                                ]
                            }, f, indent=2, ensure_ascii=False)
                    
                    results[img_path.name] = result_info
                    
                except Exception as e:
                    self.logger.error(f"处理图片 {img_path} 时出错: {e}")
                
                pbar.update(1)
        
        pbar.close()
        return results

    def compute_uncertainty(self, results: Dict[str, Dict], 
                          score_thr: float = 0.09) -> Dict[str, Dict]:
        """计算不确定性（在推理完成后调用）"""
        # self.logger.info("开始计算不确定性...")
        # print("开始计算不确定性...")
        
        processed_results = {}
        for img_name, result_info in tqdm(results.items(), desc="计算不确定性"):
            try:
                result = result_info['result']
                
                # 使用UncertaintyMetrics类计算不确定性
                uncertainty = self.uncertainty_metrics.compute_uncertainty(
                    result,
                    methods=self.uncertainty_methods,
                    min_score_thresh=score_thr
                )
                
                # 更新结果
                processed_results[img_name] = {
                    'result': result,
                    'vis_path': result_info.get('vis_path'),
                    'uncertainty': uncertainty
                }
                
            except Exception as e:
                self.logger.error(f"计算不确定性时出错 ({img_name}): {str(e)}")
                processed_results[img_name] = result_info
        
        # 保存不确定性结果
        uncertainty_path = self.uncertainty_dir / 'uncertainty_results.json'
        with open(uncertainty_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': self.timestamp,
                'uncertainty_methods': self.uncertainty_methods,
                'results': {
                    img_name: {
                        'uncertainty': info['uncertainty']
                    } for img_name, info in processed_results.items()
                    if 'uncertainty' in info
                }
            }, f, indent=2, ensure_ascii=False)
        
        # self.logger.info(f"不确定性结果已保存至: {uncertainty_path}")
        # print(f"不确定性结果已保存至: {uncertainty_path}")
        return processed_results


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='SUAL 简化版检测推理')
    
    # 基础参数
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='模型文件路径')
    parser.add_argument('input', help='输入图片路径或目录')
    parser.add_argument('--device', default='cuda:0', help='设备')
    parser.add_argument('--batch-size', type=int, default=4, help='批处理大小')
    parser.add_argument('--score-thr', type=float, default=0.3, help='得分阈值')
    parser.add_argument('--output-dir', default='outputs', help='输出目录')
    parser.add_argument('--sample-size', type=int, default=0, help='采样数量')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of worker threads')
        # 添加不确定性相关参数
    parser.add_argument('--no-uncertainty', action='store_true', 
                       help='禁用不确定性分析')
    parser.add_argument('--uncertainty-methods', nargs='+',
                       choices=['basic', 'entropy', 'variance', 'quantile', 
                               'density', 'value', 'box', 'all', 'sor', 'margin',
                               'least_confident', 'ssc'],  
                       default=['all'], 
                       help='不确定性计算方法')

    args = parser.parse_args()

    try:
        # 初始化检测器
        detector = SimDetectionInference(
            args.config,
            args.checkpoint,
            device=args.device,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            score_thr=args.score_thr,
            enable_uncertainty=not args.no_uncertainty,  # 从命令行参数设置
            uncertainty_methods=args.uncertainty_methods,
        )

        # 执行推理
        results = detector.inference(
            args.input,
            sample_size=args.sample_size
        )
        # 保存整体推理结果为JSON
        results_save_path = detector.current_output_dir / 'all_results.json'
        results_to_save = {}
        for img_name, result_info in results.items():
            results_to_save[img_name] = {
                'scores': result_info['result'].pred_instances.scores.cpu().numpy().tolist(),
                'labels': result_info['result'].pred_instances.labels.cpu().numpy().tolist(),
                'bboxes': result_info['result'].pred_instances.bboxes.cpu().numpy().tolist(),
                'labels_names': [
                    detector.model.dataset_meta['classes'][label]
                    for label in result_info['result'].pred_instances.labels.cpu().numpy()
                ],
                'vis_path': result_info.get('vis_path')
            }
        
        with open(results_save_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        print(f"整体推理结果已保存至: {results_save_path}")

        # 计算不确定性
        results_with_uncertainty = detector.compute_uncertainty(
            results,
            score_thr=args.score_thr
        )
        # print(results_with_uncertainty)
        print(f"\n推理完成! 共处理 {len(results)} 张图片")
        print(f"结果保存在: {detector.output_dir}/{detector.timestamp}")
        
    except Exception as e:
        print(f"推理失败: {e}")
        raise


if __name__ == '__main__':
    main()