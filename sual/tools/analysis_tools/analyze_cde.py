# import sys
# sys.path.append('E:\\sual')
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import torch
import logging
import sys
import argparse
from mmengine.config import Config
from sual.inference.detector import DetectionInference

def setup_logger(name=None) -> logging.Logger:
    """设置日志记录器"""
    # 创建logger
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)

    # 避免重复添加handler
    if not logger.handlers:
        # 创建控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # 创建文件handler
        file_handler = logging.FileHandler('cde_analysis.log')
        file_handler.setLevel(logging.INFO)

        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # 添加handler到logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        # 设置propagate为False，避免重复日志
        logger.propagate = False

    return logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CDE分析工具')
    parser.add_argument('--config', type=str, required=True,
                      help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='模型检查点路径')
    parser.add_argument('--image-folder', type=str, required=True,
                      help='图片文件夹路径')
    parser.add_argument('--annotation', type=str, required=True,
                      help='标注文件路径')
    parser.add_argument('--conf-thresh', type=float, default=0.3,
                      help='置信度阈值 (default: 0.3)')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                      help='IoU阈值 (default: 0.5)')
    parser.add_argument('--output', type=str, default='cde_analysis_results.json',
                      help='输出结果文件路径 (default: cde_analysis_results.json)')
    return parser.parse_args()

class CDEAnalyzer:
    def __init__(
            self,
            config_file: str,
            checkpoint_file: str,
            conf_thresh: float = 0.3,
            iou_thresh: float = 0.5,
            logger: logging.Logger = None
    ):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.logger = logger or setup_logger()
        
        # 初始化推理器
        self.detector = DetectionInference(
            config_file=config_file,
            checkpoint_file=checkpoint_file,
            output_dir='cde_outputs',
            enable_uncertainty=False
        )
        
        # 加载配置
        self.cfg = Config.fromfile(config_file)
        
        self.logger.info(f"初始化CDEAnalyzer完成")
        self.logger.info(f"配置文件: {config_file}")
        self.logger.info(f"检查点文件: {checkpoint_file}")
        self.logger.info(f"置信度阈值: {conf_thresh}")
        self.logger.info(f"IoU阈值: {iou_thresh}")
    
    def calculate_iou(self, box1: np.ndarray, box2: List[float]) -> float:
        """计算两个边界框的IoU"""
        box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def analyze_prediction(self, pred_result: Dict, gt_boxes: List[Dict]) -> Dict[str, List[Dict]]:
        """分析单张图片的预测结果"""
        errors = {
            'high_conf_misclass': [],  # 高置信度分类错误（单类别时为0）
            'high_spatial_error': [],  # 高空间误差
            'false_negative': []       # 漏检
        }
        
        self.logger.info(f"\nGround truth boxes数量: {len(gt_boxes)}")
        
        # 检查是否为单类别预测
        is_single_class = True  # 设置为单类别标志
        
        if not pred_result or 'pred_instances' not in pred_result:
            self.logger.info("未找到预测结果，所有GT框判定为漏标")
            for gt in gt_boxes:
                errors['false_negative'].append({'gt': gt})
            return errors

        pred_instances = pred_result['pred_instances']
        pred_bboxes = pred_instances.bboxes.cpu().numpy()
        pred_scores = pred_instances.scores.cpu().numpy()
        pred_labels = pred_instances.labels.cpu().numpy()

        # 计算平均置信度和标准差
        if len(pred_scores) > 0:
            mean_conf = np.mean(pred_scores)
            std_conf = np.std(pred_scores)
            self.logger.info(f"平均置信度: {mean_conf:.3f}")
            self.logger.info(f"置信度标准差: {std_conf:.3f}")
        else:
            mean_conf = std_conf = 0
            self.logger.info("没有预测框")

        # 对每个GT框进行分析
        for gt_idx, gt in enumerate(gt_boxes):
            self.logger.info(f"\n分析第 {gt_idx+1} 个ground truth box:")
            gt_bbox = gt['bbox']
            best_match = {
                'iou': 0,
                'pred_idx': -1,
                'score': 0
            }

            # 遍历所有预测框找到最佳匹配
            for pred_idx, (bbox, score, label) in enumerate(zip(pred_bboxes, pred_scores, pred_labels)):
                iou = self.calculate_iou(bbox, gt_bbox)
                
                if iou > best_match['iou']:
                    best_match = {
                        'iou': iou,
                        'pred_idx': pred_idx,
                        'bbox': bbox,
                        'score': score,
                        'label': label
                    }

            # 根据CDE定义判断错误类型
            if best_match['pred_idx'] >= 0:
                score = best_match['score']
                iou = best_match['iou']
                bbox = best_match['bbox']
                label = best_match['label']

                # 1. 高置信度错分类（仅在多类别情况下判断）
                if not is_single_class:
                    if (iou >= 0.5 and 
                        score > mean_conf + 2 * std_conf and 
                        label != gt['category_id'] - 1):
                        errors['high_conf_misclass'].append({
                            'pred': {
                                'bbox': bbox.tolist(),
                                'score': float(score),
                                'label': int(label)
                            },
                            'gt': gt,
                            'iou': float(iou)
                        })
                        self.logger.info(f"判定为高置信度错分类 (IoU: {iou:.3f}, 置信度: {score:.3f})")

                # 2. 高空间误差
                if score > 0.3 and iou < 0.5:
                    errors['high_spatial_error'].append({
                        'pred': {
                            'bbox': bbox.tolist(),
                            'score': float(score),
                            'label': int(label)
                        },
                        'gt': gt,
                        'iou': float(iou)
                    })
                    self.logger.info(f"判定为高空间误差 (IoU: {iou:.3f}, 置信度: {score:.3f})")

            # 3. 漏检
            if best_match['pred_idx'] < 0 or best_match['iou'] < 0.1:
                errors['false_negative'].append({
                    'gt': gt,
                    'best_iou': float(best_match['iou']) if best_match['pred_idx'] >= 0 else 0,
                    'reason': 'no_predictions' if best_match['pred_idx'] < 0 else 'low_iou'
                })
                self.logger.info("判定为漏检")

        return errors




    # def analyze_prediction(self, pred_result: Dict, gt_boxes: List[Dict]) -> Dict[str, List[Dict]]:
    #     """分析单张图片的预测结果"""
    #     errors = {
    #         'high_conf_misclass': [],
    #         'high_spatial_error': [],
    #         'false_negative': []
    #     }
        
    #     self.logger.info(f"\nGround truth boxes数量: {len(gt_boxes)}")
        
    #     if not pred_result or 'pred_instances' not in pred_result:
    #         self.logger.info("未找到预测结果，所有GT框判定为漏标")
    #         for gt in gt_boxes:
    #             errors['false_negative'].append({'gt': gt})
    #         return errors

    #     pred_instances = pred_result['pred_instances']
    #     pred_bboxes = pred_instances.bboxes.cpu().numpy()
    #     pred_scores = pred_instances.scores.cpu().numpy()
    #     pred_labels = pred_instances.labels.cpu().numpy()

    #     # 筛选置信度大于阈值的框
    #     confidence_threshold = 0.7
    #     indices = pred_scores > confidence_threshold

    #     # 使用筛选后的预测结果
    #     filtered_bboxes = pred_bboxes[indices]
    #     filtered_scores = pred_scores[indices]
    #     filtered_labels = pred_labels[indices]

    #     self.logger.info(f"原始预测框数量: {len(pred_bboxes)}")
    #     self.logger.info(f"筛选后预测框数量: {len(filtered_bboxes)}")
    #     self.logger.info(f"筛选后预测得分数量: {len(filtered_scores)}")
    #     self.logger.info(f"筛选后预测标签数量: {len(filtered_labels)}")

    #     if len(filtered_scores) > 0:
    #         mean_conf = np.mean(filtered_scores)
    #         std_conf = np.std(filtered_scores)
    #         self.logger.info(f"平均置信度: {mean_conf:.3f}")
    #         self.logger.info(f"置信度标准差: {std_conf:.3f}")
    #     else:
    #         mean_conf = std_conf = 0
    #         self.logger.info("没有高置信度的预测框")
        
    #     for gt_idx, gt in enumerate(gt_boxes):
    #         self.logger.info(f"\n分析第 {gt_idx+1} 个ground truth box:")
    #         self.logger.info(f"GT box: {gt['bbox']}")
    #         self.logger.info(f"GT category: {gt['category_id']}")
            
    #         gt_bbox = gt['bbox']
    #         gt_area = gt_bbox[2] * gt_bbox[3]
    #         best_iou = 0
    #         best_pred_idx = -1
    #         is_completely_contained = False
            
    #         # 如果筛选后没有预测框，直接判定为漏标
    #         if len(filtered_bboxes) == 0:
    #             errors['false_negative'].append({
    #                 'gt': gt,
    #                 'reason': 'no_predictions'
    #             })
    #             self.logger.info("没有预测框，判定为漏标")
    #             continue

    #         # 在匹配预测框的循环中添加面积限制检查
    #         for pred_idx, (bbox, score, label) in enumerate(zip(filtered_bboxes, filtered_scores, filtered_labels)):
    #             iou = self.calculate_iou(bbox, gt_bbox)
    #             pred_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                
    #             # 计算面积比例
    #             area_ratio = pred_area / gt_area
                
    #             # 如果预测框面积超过真实框的1.5倍，跳过这个预测框
    #             if area_ratio > 1.3:
    #                 self.logger.info(f"预测框 {pred_idx} 面积过大 (比例: {area_ratio:.2f})，跳过")
    #                 continue
                
    #             self.logger.info(f"与预测框 {pred_idx} 的IoU: {iou:.3f}, 置信度: {score:.3f}, 面积比例: {area_ratio:.2f}")
                
    #             gt_box_xyxy = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]
    #             is_contained = (bbox[0] <= gt_box_xyxy[0] and bbox[1] <= gt_box_xyxy[1] and
    #                         bbox[2] >= gt_box_xyxy[2] and bbox[3] >= gt_box_xyxy[3])
                
    #             if is_contained:
    #                 is_completely_contained = True
    #                 if area_ratio <= 1.3:  # 添加面积限制
    #                     best_iou = iou
    #                     best_pred_idx = pred_idx
    #                     self.logger.info(f"找到完全包含且面积合适的预测框 {pred_idx}")
    #                     break
                
    #             if iou > best_iou and area_ratio <= 1.5:  # 添加面积限制
    #                 best_iou = iou
    #                 best_pred_idx = pred_idx
    #                 self.logger.info(f"更新最佳IoU: {best_iou:.3f}")

    #         # 在判断错误类型时也考虑面积比例
    #         if best_pred_idx >= 0:
    #             bbox = filtered_bboxes[best_pred_idx]
    #             score = filtered_scores[best_pred_idx]
    #             label = filtered_labels[best_pred_idx]
    #             pred_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    #             area_ratio = pred_area / gt_area
                
    #             if best_iou >= 0.9 and area_ratio <= 1.5:  # 添加面积限制
    #                 if label != gt['category_id'] - 1 and score > mean_conf + 2 * std_conf:
    #                     errors['high_conf_misclass'].append({
    #                         'pred': {
    #                             'bbox': bbox.tolist(),
    #                             'score': float(score),
    #                             'label': int(label),
    #                             'area_ratio': float(area_ratio)
    #                         },
    #                         'gt': gt,
    #                         'iou': float(best_iou)
    #                     })
    #                     self.logger.info("判定为高置信度错分类")
    #             elif 0.1 <= best_iou < 0.5 and area_ratio <= 1.5:  # 添加面积限制
    #                 errors['high_spatial_error'].append({
    #                     'pred': {
    #                         'bbox': bbox.tolist(),
    #                         'score': float(score),
    #                         'label': int(label),
    #                         'area_ratio': float(area_ratio)
    #                     },
    #                     'gt': gt,
    #                     'iou': float(best_iou)
    #                 })
    #                 self.logger.info("判定为空间定位错误")

    #         # 修改漏标判定条件
    #         if best_iou < 0.1 or best_pred_idx < 0 or (best_pred_idx >= 0 and pred_area / gt_area < 1.5):
    #             errors['false_negative'].append({
    #                 'gt': gt,
    #                 'best_iou': float(best_iou),
    #                 'reason': 'low_iou' if best_iou < 0.1 else 'area_mismatch'
    #             })
    #             self.logger.info(f"判定为漏标 (IoU: {best_iou:.3f}, 面积比例: {pred_area/gt_area if best_pred_idx >= 0 else 'N/A'})")
            
    #         # # 使用筛选后的预测框进行匹配
    #         # for pred_idx, (bbox, score, label) in enumerate(zip(filtered_bboxes, filtered_scores, filtered_labels)):
    #         #     iou = self.calculate_iou(bbox, gt_bbox)
    #         #     pred_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                
    #         #     self.logger.info(f"与预测框 {pred_idx} 的IoU: {iou:.3f}, 置信度: {score:.3f}")
                
    #         #     gt_box_xyxy = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]
    #         #     is_contained = (bbox[0] <= gt_box_xyxy[0] and bbox[1] <= gt_box_xyxy[1] and
    #         #                   bbox[2] >= gt_box_xyxy[2] and bbox[3] >= gt_box_xyxy[3])
                
    #         #     if is_contained:
    #         #         is_completely_contained = True
    #         #         if pred_area <= gt_area * 1.25:
    #         #             best_iou = iou
    #         #             best_pred_idx = pred_idx
    #         #             self.logger.info(f"找到完全包含的预测框 {pred_idx}")
    #         #             break
                
    #         #     if iou > best_iou:
    #         #         best_iou = iou
    #         #         best_pred_idx = pred_idx
    #         #         self.logger.info(f"更新最佳IoU: {best_iou:.3f}")
            
    #         # self.logger.info(f"最终最佳IoU: {best_iou:.3f}")
            
    #         # if best_pred_idx >= 0:
    #         #     bbox = filtered_bboxes[best_pred_idx]
    #         #     score = filtered_scores[best_pred_idx]
    #         #     label = filtered_labels[best_pred_idx]
    #         #     pred_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                
    #         #     if best_iou >= 0.9:
    #         #         if label != gt['category_id'] - 1 and score > mean_conf + 2 * std_conf:
    #         #             errors['high_conf_misclass'].append({
    #         #                 'pred': {
    #         #                     'bbox': bbox.tolist(),
    #         #                     'score': float(score),
    #         #                     'label': int(label)
    #         #                 },
    #         #                 'gt': gt,
    #         #                 'iou': float(best_iou)
    #         #             })
    #         #             self.logger.info("判定为高置信度错分类")
    #         #     elif 0.1 <= best_iou < 0.5:
    #         #         errors['high_spatial_error'].append({
    #         #             'pred': {
    #         #                 'bbox': bbox.tolist(),
    #         #                 'score': float(score),
    #         #                 'label': int(label)
    #         #             },
    #         #             'gt': gt,
    #         #             'iou': float(best_iou)
    #         #         })
    #         #         self.logger.info("判定为空间定位错误")
            
    #         # if best_iou < 0.1 or (best_pred_idx >= 0 and pred_area > gt_area * 1.25 and not is_completely_contained):
    #         #     errors['false_negative'].append({
    #         #         'gt': gt,
    #         #         'best_iou': float(best_iou),
    #         #         'reason': 'low_iou' if best_iou < 0.1 else 'area_mismatch'
    #         #     })
    #         #     self.logger.info("判定为漏标")
        
    #     return errors

    def analyze_folder(self, image_folder: str, annotation_path: str, output_path: str):
        """分析文件夹中的所有图片"""
        self.logger.info(f"开始分析文件夹: {image_folder}")
        
        # 加载标注文件
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        
        self.logger.info(f"加载标注文件: {annotation_path}")
        self.logger.info(f"总标注数量: {len(annotations['annotations'])}")
        
        # 创建图片ID到标注的映射
        image_id_to_annotations = {}
        for ann in annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in image_id_to_annotations:
                image_id_to_annotations[img_id] = []
            image_id_to_annotations[img_id].append(ann)
        
        # 创建图片文件名到图片ID的映射
        filename_to_image_id = {
            img['file_name']: img['id']
            for img in annotations['images']
        }
        
        # 获取文件夹中的所有图片
        image_folder = Path(image_folder)
        image_files = list(image_folder.glob('*.jpg')) + list(image_folder.glob('*.png'))
        
        if not image_files:
            self.logger.warning(f"在 {image_folder} 中未找到图片文件")
            return
        
        self.logger.info(f"找到 {len(image_files)} 张图片")
        
        # 进行推理
        inference_results = self.detector.inference(str(image_folder))
        
        # 用于累计所有图片的错误
        total_errors = {
            'high_conf_misclass': [],
            'high_spatial_error': [],
            'false_negative': []
        }
        
        # 处理每张图片的结果
        for image_path in image_files:
            image_name = image_path.name
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"处理图片: {image_name}")
            
            # 获取图片ID和对应的标注
            if image_name in filename_to_image_id:
                img_id = filename_to_image_id[image_name]
                image_annotations = image_id_to_annotations.get(img_id, [])
                self.logger.info(f"图片ID: {img_id}")
                self.logger.info(f"找到 {len(image_annotations)} 个标注")
            else:
                self.logger.warning(f"警告：在标注文件中未找到图片 {image_name} 的信息")
                image_annotations = []
            
            if image_name in inference_results:
                det_data_sample = inference_results[image_name]['result']
                pred_instances = det_data_sample.pred_instances
                
                # 分析推理结果
                errors = self.analyze_prediction(
                    {'pred_instances': pred_instances}, 
                    image_annotations
                )
                
                # 累计错误
                for error_type in total_errors:
                    total_errors[error_type].extend(errors[error_type])
                    
                # 打印当前图片的分析结果
                self.logger.info(f"\n图片 {image_name} 的分析结果:")
                for error_type, error_list in errors.items():
                    self.logger.info(f"{error_type}: {len(error_list)}")
            else:
                self.logger.warning(f"推理结果中未找到图像 {image_name}")
        
        # 打印总体统计信息
        self.logger.info("\n" + "="*50)
        self.logger.info("总体分析结果:")
        for error_type, error_list in total_errors.items():
            self.logger.info(f"{error_type}: {len(error_list)} 个错误")
            
        # 保存分析结果
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(total_errors, f, indent=2, ensure_ascii=False)
        self.logger.info(f"分析结果已保存至: {output_path}")

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志记录器
    logger = setup_logger('CDE_Analysis')
    
    try:
        # 创建分析器实例
        analyzer = CDEAnalyzer(
            config_file=args.config,
            checkpoint_file=args.checkpoint,
            conf_thresh=args.conf_thresh,
            iou_thresh=args.iou_thresh,
            logger=logger
        )
        
        # 执行分析
        analyzer.analyze_folder(
            image_folder=args.image_folder,
            annotation_path=args.annotation,
            output_path=args.output
        )
    except Exception as e:
        logger.error(f"发生错误: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()