import argparse
import json
import random
import shutil
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import os

def setup_logger():
    """设置日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='准备监督学习数据集（从COCO数据集中随机采样）'
    )
    parser.add_argument(
        '--source-root', 
        type=str, 
        required=True,
        help='源COCO数据集根目录'
    )
    parser.add_argument(
        '--output-root', 
        type=str, 
        required=True,
        help='输出数据集的根目录'
    )
    parser.add_argument(
        '--num-images', 
        type=int, 
        required=True,
        help='要采样的总图片数量'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='随机种子'
    )
    return parser.parse_args()

def load_coco_annotations(ann_file: str) -> Dict:
    """加载COCO标注文件"""
    with open(ann_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def random_split_images(
    images: List[Dict],
    num_total: int,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """随机划分图片为训练集和验证集"""
    random.seed(seed)
    
    # 随机采样指定数量的图片
    selected_images = random.sample(images, min(num_total, len(images)))
    
    # 计算训练集大小
    num_train = int(len(selected_images) * train_ratio)
    
    # 随机打乱
    random.shuffle(selected_images)
    
    # 划分训练集和验证集
    train_images = selected_images[:num_train]
    val_images = selected_images[num_train:]
    
    return train_images, val_images

def get_annotations_for_images(
    annotations: List[Dict],
    image_ids: List[int]
) -> List[Dict]:
    """获取指定图片ID的标注信息"""
    return [ann for ann in annotations if ann['image_id'] in image_ids]

def create_dataset_structure(output_root: Path):
    """创建数据集目录结构"""
    # 创建主目录
    (output_root / 'train2017').mkdir(parents=True, exist_ok=True)
    (output_root / 'val2017').mkdir(parents=True, exist_ok=True)
    (output_root / 'annotations').mkdir(parents=True, exist_ok=True)

def copy_images(
    images: List[Dict],
    src_dir: Path,
    dst_dir: Path,
    logger: logging.Logger
):
    """复制图片文件"""
    for img in images:
        src_path = src_dir / img['file_name']
        dst_path = dst_dir / img['file_name']
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            logger.error(f"复制图片失败 {img['file_name']}: {str(e)}")

def save_annotations(
    annotations: Dict,
    output_file: Path,
    logger: logging.Logger
):
    """保存标注文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"保存标注文件失败 {output_file}: {str(e)}")

def main():
    args = parse_args()
    logger = setup_logger()
    
    # 设置路径
    source_root = Path(args.source_root)
    output_root = Path(args.output_root) / f'coco_{args.num_images}'
    
    # 创建数据集目录结构
    create_dataset_structure(output_root)
    
    # 加载原始标注文件
    train_ann = load_coco_annotations(
        source_root / 'annotations/instances_train2017.json'
    )
    
    # 随机划分图片
    train_images, val_images = random_split_images(
        train_ann['images'],
        args.num_images,
        seed=args.seed
    )
    
    # 获取训练集和验证集的图片ID
    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}
    
    # 获取对应的标注
    train_annotations = get_annotations_for_images(
        train_ann['annotations'],
        train_image_ids
    )
    val_annotations = get_annotations_for_images(
        train_ann['annotations'],
        val_image_ids
    )
    
    # 创建新的标注文件
    new_train_ann = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': train_ann['categories']
    }
    
    new_val_ann = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': train_ann['categories']
    }
    
    # 复制图片
    logger.info("复制训练集图片...")
    copy_images(
        train_images,
        source_root / 'train2017',
        output_root / 'train2017',
        logger
    )
    
    logger.info("复制验证集图片...")
    copy_images(
        val_images,
        source_root / 'train2017',
        output_root / 'val2017',
        logger
    )
    
    # 保存新的标注文件
    logger.info("保存标注文件...")
    save_annotations(
        new_train_ann,
        output_root / 'annotations/instances_train2017.json',
        logger
    )
    save_annotations(
        new_val_ann,
        output_root / 'annotations/instances_val2017.json',
        logger
    )
    
    # 打印统计信息
    logger.info(f"\n数据集创建完成: {output_root}")
    logger.info(f"训练集图片数量: {len(train_images)}")
    logger.info(f"训练集标注数量: {len(train_annotations)}")
    logger.info(f"验证集图片数量: {len(val_images)}")
    logger.info(f"验证集标注数量: {len(val_annotations)}")

if __name__ == '__main__':
    main()
    
    
    
    
    # python prepare_supervised_learning_dataset.py \
    # --source-root /path/to/source/coco \
    # --output-root /path/to/output \
    # --num-images 1000 \
    # --seed 42