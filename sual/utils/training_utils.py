
from typing import Optional
from pathlib import Path
import re
from mmengine.logging import MMLogger
from typing import List


"""Training utility functions.

This module contains helper functions for model training process, including:
- Checkpoint management and finding
- Training hyperparameter adjustment
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
    max_per_img_init, max_per_img_final = 300, 200
    iou_threshold_init, iou_threshold_final = 0.5, 0.4
    
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