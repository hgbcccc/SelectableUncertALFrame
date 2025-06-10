# # # sual/inference/detector.py

import warnings

warnings.filterwarnings("ignore")

# 此处开始写你的主要代码逻辑，后续代码执行中所有警告都会被忽略
import os
import random 
import json
import pickle
import logging
import asyncio
import psutil
from pathlib import Path
from datetime import datetime
from typing import Union, Tuple, List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import mmcv
import torch
import numpy as np
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample
from..core.uncertainty.metrics import UncertaintyMetrics
import argparse
import numpy as np


class DetectionInference:
    """增强版目标检测推理器"""
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def __init__(
            self,
            config_file: str,
            checkpoint_file: str,
            device: str = 'cuda:0',
            output_dir: str = 'outputs',
            enable_uncertainty: bool = True,
            use_fp16: bool = False,
            batch_size: int = 4,
            max_memory_usage: float = 0.8,
            cache_size: int = 1000,
            num_workers: int = 4,
            score_thr: float = 0.4,
            enable_mc_dropout: bool = False,  # 保持原有参数名
            mc_dropout_times: int = 20,  # 保持原有参数名
            uncertainty_methods: Optional[List[str]] = None,  # 新增参数，不确定性分析方法
            sample_size: int = 0, # 新增参数：采样数量，0表示使用所有图片
    ):
        """初始化检测器

        Args:
            config_file: 配置文件路径
            checkpoint_file: 模型文件路径
            device: 设备
            output_dir: 输出目录
            enable_uncertainty: 是否启用不确定性分析
            use_fp16: 是否使用半精度
            batch_size: 批处理大小
            max_memory_usage: 最大内存使用率
            cache_size: 结果缓存大小
            num_workers: 线程池大小
            score_thr: 得分阈值
            enable_mc_dropout: 是否启用MC Dropout
            mc_dropout_times: MC Dropout次数
        """
        import nest_asyncio
        nest_asyncio.apply()
        # 注册模块
        register_all_modules()
        
        # 基础配置
        self.enable_uncertainty = enable_uncertainty
        self.device = device
        self.batch_size = batch_size
        self.max_memory_usage = max_memory_usage
        self.use_fp16 = use_fp16
        
        # 初始化模型
        cfg_options = {'fp16': use_fp16} if use_fp16 else {}
        self.model = init_detector(config_file, checkpoint_file, device=device, cfg_options=cfg_options)
        
        # 设置输出目录
        self._setup_directories(output_dir)

        # 设置日志
        self._setup_logging()

        # 初始化缓存和线程池
        self.cache = {}
        self.cache_size = cache_size
        self.thread_pool = ThreadPoolExecutor(max_workers=num_workers)

        # MC Dropout 配置
        self.score_thr = score_thr
        self.enable_mc_dropout = enable_mc_dropout
        self.mc_dropout_times = mc_dropout_times
        
        # 如果启用 MC Dropout，设置相关层
        if self.enable_mc_dropout:
            self.logger.info("启用 MC Dropout")
            self.model.train()  # 设置为训练模式
            self._enable_dropout()  # 启用所有dropout层
            
            # 检查是否有dropout层
            dropout_layers = []
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)) or 'Dropout' in module.__class__.__name__:
                    dropout_layers.append(name)
            
            if dropout_layers:
                self.logger.info(f"找到 {len(dropout_layers)} 个 Dropout 层:")
                for layer in dropout_layers:
                    self.logger.info(f"- {layer}")
            else:
                self.logger.warning("没有找到任何 Dropout 层，MC Dropout 可能无效")
        
        # 不确定性分析配置
        if enable_uncertainty:
            self.uncertainty_metrics = UncertaintyMetrics()
            
        # 初始化可视化器
        self._setup_visualizer()
        self.sample_size = sample_size

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
        print(f"enable_uncertainty value: {self.enable_uncertainty}")  # 添加日志输出查看值
        if self.enable_uncertainty:
            self.uncertainty_dir = self.current_output_dir / 'uncertainty'
            self.uncertainty_dir.mkdir(exist_ok=True)

    def _check_memory(self):
        """检查内存使用情况"""
        memory_percent = psutil.virtual_memory().percent / 100
        if memory_percent > self.max_memory_usage:
            self.logger.warning("Memory usage high, triggering garbage collection")
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    
    def _setup_logging(self):
        """配置日志系统"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 文件处理器
        fh = logging.FileHandler(self.current_output_dir / 'inference.log')
        fh.setLevel(logging.INFO)

        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _setup_visualizer(self):
        """配置可视化器"""
        cfg = self.model.cfg.copy()
        cfg.visualizer.save_dir = str(self.vis_dir)
        self.visualizer = VISUALIZERS.build(cfg.visualizer)
        self.visualizer.dataset_meta = self.model.dataset_meta

    @contextmanager
    def error_handling(self, operation: str):
        """错误处理上下文管理器"""
        try:
            yield
        except Exception as e:
            self.logger.error(f"{operation} failed: {str(e)}")
            raise

    def _get_from_cache(self, img_path: Path) -> Optional[Dict]:
        """从缓存中获取结果"""
        cache_key = str(img_path)
        if cache_key in self.cache:
            self.logger.debug(f"Cache hit for {img_path}")
            return self.cache[cache_key]
        return None

    def _update_cache(self, img_path: Path, result: Dict):
        """更新缓存"""
        if len(self.cache) >= self.cache_size:
            # 使用FIFO策略清理缓存
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            self.logger.debug(f"Cache full, removed {oldest_key}")
        self.cache[str(img_path)] = result

    def _get_image_files(self, input_path: Union[str, Path]) -> List[Path]:
        """获取所有支持的图片文件路径"""
        with self.error_handling("Getting image files"):
            input_path = Path(input_path)
            if input_path.is_file():
                if input_path.suffix.lower() in self.SUPPORTED_FORMATS:
                    return [input_path]
                raise ValueError(f"Unsupported file format: {input_path}")
            image_files = []
            for fmt in self.SUPPORTED_FORMATS:
                image_files.extend(input_path.glob(f"*{fmt.lower()}"))
            return sorted(image_files)

    async def _load_image(self, img_path: Path) -> Optional[np.ndarray]:
        """异步加载图片"""
        try:
            img = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, mmcv.imread, str(img_path))
            if img is None:
                self.logger.warning(f"Failed to read image: {img_path}")
                return None
            return img
        except Exception as e:
            self.logger.error(f"Error loading image {img_path}: {e}")
            return None

##########上##########新添加的MC Dropout实现代码#############################
    def _enable_dropout(self):
        """启用模型中的所有dropout层"""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()  # 设置为训练模式以启用dropout
            elif 'Dropout' in module.__class__.__name__:  # 处理自定义dropout层
                module.train()
            # 禁用BN层的更新
            elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                module.eval()

    async def _mc_forward(self, img: np.ndarray) -> List[Dict]:
        """执行多次MC Dropout前向传播，并获取所有类别的概率"""
        mc_results = []
        
        for i in range(self.mc_dropout_times):  # 使用原版参数名
            # 执行单次前向传播，但获取原始输出
            result = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: self._forward_with_all_probs(img)
            )
            
            # 从结果中提取信息
            bboxes = result.pred_instances.bboxes.cpu().numpy()
            scores = result.pred_instances.scores.cpu().numpy()
            labels = result.pred_instances.labels.cpu().numpy()
            all_class_probs = result.pred_instances.all_class_probs.cpu().numpy()
            
            # 保存这次采样的结果
            sample_result = {
                'sample_id': i,
                'bboxes': bboxes.tolist(),
                'scores': scores.tolist(),
                'labels': labels.tolist(),
                'labels_names': [self.model.dataset_meta['classes'][label] for label in labels],
                'all_class_probs': all_class_probs.tolist()
            }
            mc_results.append(sample_result)
        
        return mc_results

    def _forward_with_all_probs(self, img: np.ndarray) -> DetDataSample:
        """执行前向传播并获取所有类别的概率"""
        # 获取模型的分类头输出
        with torch.no_grad():
            feat = self.model.extract_feat(img)
            results_list = self.model.bbox_head.predict(feat, None, rescale=True)
            
            # 获取原始的分类分数（logits）
            cls_scores = results_list[0].scores  # [num_boxes, num_classes]
            
            # 使用softmax转换为概率
            all_class_probs = F.softmax(cls_scores, dim=-1)
            
            # 创建DetDataSample并添加所有类别的概率
            results = self.model.bbox_head.predict_by_feat(
                *self.model.bbox_head.predict(feat, None, rescale=True),
                rescale=True,
                with_nms=True
            )
            results[0].pred_instances.all_class_probs = all_class_probs
            
            return results[0]

    def _aggregate_mc_results(self, mc_results: List[Dict]) -> Dict:
        """聚合多次MC Dropout的结果"""
        # 将所有采样结果转换为numpy数组
        all_bboxes = [np.array(sample['bboxes']) for sample in mc_results]
        all_scores = [np.array(sample['scores']) for sample in mc_results]
        all_labels = [np.array(sample['labels']) for sample in mc_results]
        
        # 计算统计量
        mean_bboxes = np.mean(all_bboxes, axis=0)
        std_bboxes = np.std(all_bboxes, axis=0)
        mean_scores = np.mean(all_scores, axis=0)
        std_scores = np.std(all_scores, axis=0)
        mode_labels = stats.mode(all_labels, axis=0)[0]
        
        return {
            'mean_bboxes': mean_bboxes.tolist(),  # 转换为list以便JSON序列化
            'std_bboxes': std_bboxes.tolist(),
            'mean_scores': mean_scores.tolist(),
            'std_scores': std_scores.tolist(),
            'mode_labels': mode_labels.tolist(),
            'raw_samples': mc_results  # 保存原始采样结果
        }
#############下#######新添加的MC Dropout实现代码#############################




    def _preprocess_batch(self, batch_imgs: List[np.ndarray]) -> List[np.ndarray]:
        """预处理批量图片"""
        processed_imgs = []
        for img in batch_imgs:
            # 添加图片预处理逻辑，例如调整大小、标准化等
            if max(img.shape[:2]) > 2048:  # 限制最大边长
                scale = 2048 / max(img.shape[:2])
                img = mmcv.imresize(
                    img,
                    (int(img.shape[1] * scale), int(img.shape[0] * scale))
                )
            processed_imgs.append(img)
        return processed_imgs


    async def _process_batch(
        self,
        batch_imgs: List[np.ndarray],
        batch_paths: List[Path],
        score_thr: float,
        save_vis: bool,
        vis_scale: float,
        uncertainty_methods: Optional[List[str]],
        total_images: int = 0,
        processed_count: int = 0
    ) -> List[Tuple[str, Dict]]:
        """处理一批图片"""
        with self.error_handling("Processing batch"):
            batch_imgs = self._preprocess_batch(batch_imgs)
            
            batch_outputs = []
            for img_idx, (img, path) in enumerate(zip(batch_imgs, batch_paths)):
                try:
                    if self.enable_mc_dropout:
                        # 启用dropout层
                        self._enable_dropout()
                        
                        # 执行MC Dropout多次前向传播
                        mc_results = await self._mc_forward(img)
                        
                        # 聚合MC Dropout结果
                        aggregated_result = self._aggregate_mc_results(mc_results)
                        
                        result_info = {
                            'result': {
                                'mc_results': mc_results,  # 保存原始MC采样结果
                                'aggregated': aggregated_result  # 保存聚合结果
                            },
                            'vis_path': None,
                            'uncertainty': None
                        }
                        
                        # 创建用于可视化的DetDataSample
                        result_for_vis = DetDataSample()
                        pred_instances = InstanceData()
                        pred_instances.bboxes = torch.tensor(aggregated_result['mean_bboxes'])
                        pred_instances.scores = torch.tensor(aggregated_result['mean_scores'])
                        pred_instances.labels = torch.tensor(aggregated_result['mode_labels'])
                        result_for_vis.pred_instances = pred_instances
                        
                    else:
                        # 原有的单次推理逻辑
                        self.model.eval()  # 非MC Dropout模式下使用评估模式
                        with torch.cuda.amp.autocast(enabled=self.use_fp16):
                            result = await asyncio.get_event_loop().run_in_executor(
                                self.thread_pool,
                                lambda: inference_detector(self.model, img)
                            )
                        result_info = {
                            'result': result,
                            'vis_path': None,
                            'uncertainty': None
                        }
                        result_for_vis = result

                    # 获取图片尺寸
                    img_shape = img.shape[:2]
                    
                    # 可视化
                    if save_vis:
                        vis_path = str(self.vis_dir / f"{path.stem}_vis.jpg")
                        vis_path = await self._visualize_result(
                            img, result_for_vis, vis_path, score_thr, vis_scale
                        )
                        result_info['vis_path'] = vis_path

                    # 不确定性分析
                    if self.enable_uncertainty:
                        if uncertainty_methods == ['all']:
                            current_methods = 'all'
                        else:
                            current_methods = uncertainty_methods
                        
                        uncertainty = self.uncertainty_metrics.compute_uncertainty(
                            result_for_vis,
                            methods=current_methods,
                            min_score_thresh=score_thr,
                            img_shape=img_shape,
                        )
                        result_info['uncertainty'] = uncertainty

                    batch_outputs.append((path.name, result_info))
                    self._update_cache(path, result_info)
                except Exception as e:
                    self.logger.error(f"Error processing {path}: {e}")
                    continue
            return batch_outputs
    

    async def async_inference(
            self,
            input_path: Union[str, Path],
            score_thr: float = 0.3,
            save_results: bool = True,
            save_vis: bool = True,
            selected_metric: str = "entropy", 
            vis_scale: float = 1.0,
            uncertainty_methods: Optional[List[str]] = None,
            sample_size: int = 0,  # 添加采样参数
    ) -> Dict[str, Dict]:
        """异步推理入口
        
        Args:
            input_path: 输入图片路径
            score_thr: 置信度阈值
            save_results: 是否保存结果
            save_vis: 是否保存可视化结果
            selected_metric: 选择的指标
            vis_scale: 可视化缩放比例
            uncertainty_methods: 不确定性计算方法列表
            sample_size: 采样数量，0表示使用所有图片
        """
        with self.error_handling("Async inference"):
            image_files = self._get_image_files(input_path)
            self.logger.info(f"Found {len(image_files)} images")
            
            if not image_files:
                raise ValueError(f"No supported image files found in: {input_path}")

            # 处理采样逻辑
            if sample_size > 0:
                total_images = len(image_files)
                actual_sample_size = min(sample_size, total_images)
                if actual_sample_size < total_images:
                    self.logger.info(f"采样 {actual_sample_size} 张图片进行推理 (总共 {total_images} 张)")
                    # 随机采样
                    image_files = random.sample(image_files, actual_sample_size)

            results = {}
            all_results = []  # 用于不确定性排序

            # 使用tqdm创建进度条
            pbar = tqdm(total=len(image_files), desc="Processing images")

            # 批量处理图片
            processed_count = 0
            total_images = len(image_files)
            
            for i in range(0, len(image_files), self.batch_size):
                batch_paths = image_files[i:i + self.batch_size]

                # 检查缓存
                batch_to_process = []
                batch_paths_to_process = []
                for path in batch_paths:
                    cached_result = self._get_from_cache(path)
                    if cached_result is not None:
                        results[path.name] = cached_result
                        pbar.update(1)
                    else:
                        batch_paths_to_process.append(path)
                        
                if not batch_paths_to_process:
                    continue
                    
                # 异步加载图片
                batch_imgs = await asyncio.gather(*[
                    self._load_image(path) for path in batch_paths_to_process
                ])
                batch_imgs = [img for img in batch_imgs if img is not None]
                if not batch_imgs:
                    continue

                # 检查内存使用
                self._check_memory()

                # 处理批次
                batch_outputs = await self._process_batch(
                    batch_imgs,
                    batch_paths_to_process,
                    score_thr=score_thr,
                    save_vis=save_vis,
                    vis_scale=vis_scale,
                    uncertainty_methods=uncertainty_methods,
                    total_images=total_images,
                    processed_count=processed_count
                )
                
                # 更新处理计数
                processed_count += len(batch_paths_to_process)

                # 保存结果
                for img_name, result_info in batch_outputs:
                    results[img_name] = result_info
                    if save_results:
                        result = result_info['result']
                        json_path = self.results_dir / f"{Path(img_name).stem}_result.json"

                        async def save_json():
                            with open(json_path, 'w', encoding='utf-8') as f:
                                if self.enable_mc_dropout:
                                    # MC Dropout结果保存
                                    mc_data = {
                                        'mc_results': result['mc_results'],  # 原始MC采样结果
                                        'aggregated': {  # 聚合结果
                                            'mean_bboxes': result['aggregated']['mean_bboxes'],
                                            'std_bboxes': result['aggregated']['std_bboxes'],
                                            'mean_scores': result['aggregated']['mean_scores'],
                                            'std_scores': result['aggregated']['std_scores'],
                                            'mode_labels': result['aggregated']['mode_labels'].tolist()
                                        }
                                    }
                                    json.dump(mc_data, f, indent=2, ensure_ascii=False)
                                else:
                                    # 原有的单次结果保存格式
                                    json.dump(
                                        {
                                            'scores': result.pred_instances.scores.cpu().numpy().tolist(),
                                            'labels': result.pred_instances.labels.cpu().numpy().tolist(),
                                            'bboxes': result.pred_instances.bboxes.cpu().numpy().tolist(),
                                            'labels_names': [
                                                self.model.dataset_meta['classes'][label]
                                                for label in result.pred_instances.labels.cpu().numpy()
                                            ]
                                        },
                                        f,
                                        indent=2,
                                        ensure_ascii=False
                                    )

                        await asyncio.gather(save_json())

                    # 对于不确定性排序，使用聚合后的结果
                    if self.enable_mc_dropout:
                        all_results.append(result_info['result']['aggregated'])
                    else:
                        all_results.append(result_info['result'])
                    pbar.update(1)
            


            pbar.close()
            self.logger.info(f"使用的不确定性方法: {uncertainty_methods}")
            
            # 确保 uncertainty_methods 不为空
            if not uncertainty_methods:
                uncertainty_methods = ['normalized_entropy']
                self.logger.warning(f"未指定不确定性方法，使用默认方法: {uncertainty_methods}")
            
            # 计算整体不确定性排序
            if self.enable_uncertainty and all_results:
                self.logger.info("开始计算不确定性排序...")
                uncertainty_scores = []
                valid_results = []
                
                for img_name, info in results.items():
                    if info.get('uncertainty'):
                        uncertainty_scores.append({
                            'image_name': img_name,
                            'scores': info['uncertainty']
                        })
                        if self.enable_mc_dropout:
                            valid_results.append(info['result']['aggregated'])
                        else:
                            valid_results.append(info['result'])
                        
                self.logger.info(f"找到 {len(valid_results)} 个有效结果")

                # 使用配置中指定的不确定性方法
                ranked_indices = self.uncertainty_metrics.rank_samples(
                    valid_results,
                    method=uncertainty_methods[0],  # 如果输入的是列表，取第一个方法 ["ssc"],获取的是"ssc"
                    selected_metric=selected_metric,
                    strategy='max',
                    min_score_thresh=score_thr
                )
                self.logger.info(f"排序后的样本数量: {len(ranked_indices)}")
                
                # 保存排序结果
                ranking_result = {
                    'timestamp': self.timestamp,
                    'uncertainty_methods': uncertainty_methods,
                    'ranking': [str(image_files[i]) for i in ranked_indices],
                    'uncertainty_scores': uncertainty_scores,  # 保存完整的不确定性度量
                    'mc_dropout_enabled': self.enable_mc_dropout  # 添加MC Dropout状态信息
                }

                ranking_path = self.uncertainty_dir / 'uncertainty_ranking.json'
                self.logger.info(f"保存排序结果到: {ranking_path}")
                
                async def save_ranking_json():
                    with open(ranking_path, 'w', encoding='utf-8') as f:
                        json.dump(ranking_result, f, indent=2, ensure_ascii=False)

                await save_ranking_json()

            return results

    def inference(self, *args, **kwargs) -> Dict[str, Dict]:
        """同步推理入口"""
        return asyncio.run(self.async_inference(*args, **kwargs))
        


    async def _visualize_result(
            self,
            img: np.ndarray,
            result: DetDataSample,
            output_path: str,
            score_thr: float = 0.3,
            vis_scale: float = 1.0
    ) -> Optional[str]:
        """异步可视化结果"""
        try:
            # 创建新的可视化器实例以避免并发问题
            visualizer = VISUALIZERS.build(self.model.cfg.visualizer)
            visualizer.dataset_meta = self.model.dataset_meta

            # 限制图片大小
            h, w = img.shape[:2]
            max_size = 1333
            if max(h, w) > max_size:
                ratio = max_size / max(h, w)
                img = mmcv.imresize(
                    img,
                    (int(w * ratio), int(h * ratio))
                )

            # 在内存中处理可视化
            visualizer.add_datasample(
                name='',
                image=img,
                data_sample=result,
                draw_gt=False,
                pred_score_thr=score_thr,
                show=False,
                wait_time=0,
                out_file=None
            )

            # 获取并缩放可视化结果
            vis_img = visualizer.get_image()
            if vis_scale!= 1.0:
                h, w = vis_img.shape[:2]
                vis_img = mmcv.imresize(
                    vis_img,
                    (int(w * vis_scale), int(h * vis_scale))
                )

            # 异步保存图片
            await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: mmcv.imwrite(vis_img, output_path)
            )
            return output_path
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            return None

def main():
    """
    命令行入口函数，用于初始化检测器并执行推理操作，处理推理过程中的异常情况，
    最后输出推理结果的相关信息（处理图片数量及结果保存路径等）。
    """
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser(description='SUAL Enhanced Detection Inference')

    # 基础参数
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('input', help='Input image path or directory')

    # 设备和性能参数
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--use-fp16', action='store_true', help='Use FP16 inference')

    # 输出控制参数
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--no-save-results', action='store_true', help='Do not save raw results')
    parser.add_argument('--no-save-vis', action='store_true', help='Do not save visualizations')
    parser.add_argument('--vis-scale', type=float, default=1.0, help='Visualization scale factor')

    # 检测参数
    parser.add_argument('--score-thr', type=float, default=0.3, help='Score threshold')
    # parser.add_argument('--nms-thr', type=float, default=0.3, help='NMS threshold')
    # parser.add_argument('--max-det', type=int, default=100, help='Maximum detections per image')

    # MC Dropout参数
    parser.add_argument('--enable-mc-dropout', action='store_true',
                       help='Enable MC Dropout during inference')
    parser.add_argument('--mc-dropout-times', type=int, default=20,
                       help='Number of MC Dropout forward passes')

    # 不确定性相关参数
    parser.add_argument('--no-uncertainty', action='store_true', 
                       help='Disable uncertainty analysis')
    parser.add_argument('--uncertainty-methods', nargs='+',
                       choices=['basic', 'entropy', 'variance', 'quantile', 
                               'density', 'value', 'box', 'all', 'sor', 'margin',
                               'least_confident', 'ssc'],  
                       default=['all'], 
                       help='Uncertainty calculation methods')
    parser.add_argument('--selected-metric', type=str, default='normalized_entropy', 
                       help='Select specific metric for uncertainty calculation')
    
    parser.add_argument('--sample-size', type=int, default=0,
                       help='采样数量 (0表示使用所有图片)')

    args = parser.parse_args()

    try:
        # 初始化检测器
        detector = DetectionInference(
            args.config,
            args.checkpoint,
            device=args.device,
            output_dir=args.output_dir,
            enable_uncertainty=not args.no_uncertainty,
            use_fp16=args.use_fp16,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            score_thr=args.score_thr,
            enable_mc_dropout=args.enable_mc_dropout,
            mc_dropout_times=args.mc_dropout_times,
            sample_size=args.sample_size
        )

        # 配置不确定性分析
        if args.no_uncertainty:
            detector.uncertainty_metrics = None
        else:
            if args.enable_mc_dropout and 'mc_dropout' not in args.uncertainty_methods:
                # 如果启用了MC Dropout但未在方法列表中，添加它
                if args.uncertainty_methods == ['all']:
                    args.uncertainty_methods = ['mc_dropout']
                else:
                    args.uncertainty_methods.append('mc_dropout')
                print(f"已启用MC Dropout，添加到不确定性方法列表中: {args.uncertainty_methods}")

        # 执行推理
        results = detector.inference(
            args.input,
            score_thr=args.score_thr,
            save_results=not args.no_save_results,
            save_vis=not args.no_save_vis,
            vis_scale=args.vis_scale,
            uncertainty_methods=args.uncertainty_methods,
            selected_metric=args.selected_metric,  # 命令行参数不是下划线是中划线请注意
            sample_size=args.sample_size
        )

        print(f"\n推理完成! 共处理 {len(results)} 张图片")
        print(f"结果保存在: {detector.output_dir}/{detector.timestamp}")
        
        # 如果启用了MC Dropout，打印额外信息
        if args.enable_mc_dropout:
            print(f"MC Dropout 已启用，执行了 {args.mc_dropout_times} 次前向传播")
            
    except Exception as e:
        print(f"推理失败: {e}")
        raise




if __name__ == '__main__':
    main()




    #################实现batch的推理#######################
# class BatchedDetectionInference:
#     """批次推理器，用于处理未标注数据集的批次统计"""
    
#     def __init__(self, inference_engine):
#         """初始化批次推理器
        
#         Args:
#             inference_engine (DetectionInference): 基础推理引擎
#         """
#         self.inference_engine = inference_engine
#         self.batch_stats = {
#             'crown_counts': [],  # 存储每张图片的边界框数量
#             'mean': None,        # 批次均值
#             'std': None          # 批次标准差
#         }
    
#     def preprocess_dataset(self, image_dir):
#         """预处理数据集，计算批次统计信息
        
#         Args:
#             image_dir (str): 图片目录路径
#         """
#         # 使用基础推理引擎进行推理
#         print(f"\n开始批次推理处理，图片目录: {image_dir}")
#         results = self.inference_engine.inference(image_dir)
#         total_images = len(results)
#         print(f"总共需要处理 {total_images} 张图片")
        
#         # 统计每张图片的边界框数量
#         processed_count = 0
#         batch_size = self.inference_engine.batch_size
#         current_batch = 1
#         batch_boxes = []
        
#         for img_name, info in results.items():
#             try:
#                 processed_count += 1
#                 if 'result' in info:
#                     if isinstance(info['result'], dict) and 'mc_results' in info['result']:
#                         # 处理MC Dropout的结果
#                         aggregated = info['result'].get('aggregated', {})
#                         if 'mean_bboxes' in aggregated:
#                             num_boxes = len(aggregated['mean_bboxes'])
#                             self.batch_stats['crown_counts'].append(num_boxes)
#                             batch_boxes.append(num_boxes)
#                     elif hasattr(info['result'], 'pred_instances'):
#                         # 处理普通推理结果
#                         num_boxes = len(info['result'].pred_instances.bboxes)
#                         self.batch_stats['crown_counts'].append(num_boxes)
#                         batch_boxes.append(num_boxes)
                
#                 # 当处理完一个批次或所有图片时，打印批次信息
#                 if len(batch_boxes) == batch_size or processed_count == total_images:
#                     batch_mean = sum(batch_boxes) / len(batch_boxes)
#                     print(f"\n批次 {current_batch} 处理完成:")
#                     print(f"  - 处理图片数: {len(batch_boxes)}")
#                     print(f"  - 批次边界框均值: {batch_mean:.2f}")
#                     print(f"  - 进度: {processed_count}/{total_images} ({(processed_count/total_images)*100:.1f}%)")
#                     batch_boxes = []
#                     current_batch += 1
                    
#             except Exception as e:
#                 print(f"Warning: Failed to process {img_name}: {str(e)}")
#                 continue
        
#         # 计算统计量
#         if self.batch_stats['crown_counts']:
#             self.batch_stats['mean'] = np.mean(self.batch_stats['crown_counts'])
#             self.batch_stats['std'] = np.std(self.batch_stats['crown_counts'])
#             # print(f"\n数据集处理完成:")
#             # print(f"  - 总处理图片数: {total_images}")
#             # print(f"  - 总批次数: {current_batch-1}")
#             # print(f"  - 数据集边界框均值: {self.batch_stats['mean']:.2f}")
#             # print(f"  - 数据集边界框标准差: {self.batch_stats['std']:.2f}")
#         else:
#             print("Warning: No valid detection results found in the dataset.")
        
#         return results
    
#     def get_batch_stats(self):
#         """获取批次统计信息
        
#         Returns:
#             dict: 包含均值和标准差的字典
#         """
#         return {
#             'mean': self.batch_stats['mean'],
#             'std': self.batch_stats['std']
#         }