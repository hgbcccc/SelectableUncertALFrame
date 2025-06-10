import json
import random
import os.path as osp
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# 准备主动学习数据集    

def find_image_in_dirs(img_name: str, search_dirs: List[Path]) -> Path:
    """在多个目录中查找图片"""
    for dir_path in search_dirs:
        img_path = dir_path / img_name
        if img_path.exists():
            return img_path
    return None

def copy_images(image_list: List[Dict], 
                src_dirs: List[str], 
                dst_dir: str):
    """复制图片到目标目录
    
    Args:
        image_list: 图片信息列表
        src_dirs: 源图片目录列表
        dst_dir: 目标图片目录
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换所有源目录为Path对象
    src_dirs = [Path(d) for d in src_dirs]
    
    # 记录成功和失败的数量
    success_count = 0
    failed_files = []
    
    for img in image_list:
        try:
            # 在所有源目录中查找图片
            src_path = find_image_in_dirs(img['file_name'], src_dirs)
            
            if src_path is None:
                print(f"警告: 找不到文件 {img['file_name']}")
                print(f"在以下目录中查找:")
                for dir_path in src_dirs:
                    print(f"  - {dir_path}")
                failed_files.append(img['file_name'])
                continue
                
            dst_path = dst_dir / img['file_name']
            shutil.copy2(src_path, dst_path)
            success_count += 1
            
        except Exception as e:
            print(f"复制文件 {img['file_name']} 时出错: {str(e)}")
            failed_files.append(img['file_name'])
    
    print(f"成功复制 {success_count} 张图片到 {dst_dir}")
    if failed_files:
        print(f"失败 {len(failed_files)} 张图片:")
        for f in failed_files[:5]:  # 只显示前5个失败的文件
            print(f"  - {f}")
        if len(failed_files) > 5:
            print(f"  ... 还有 {len(failed_files)-5} 个文件")

def merge_coco_datasets(data_root: str) -> Tuple[Dict, List[str]]:
    """合并训练集和验证集的标注文件和图片"""
    print("合并数据集...")
    
    # 规范化路径
    data_root = Path(data_root).resolve()  # 获取绝对路径
    train_ann = data_root / 'annotations/instances_train2024.json'
    val_ann = data_root / 'annotations/instances_val2024.json'
    train_img_dir = data_root / 'train2024'
    val_img_dir = data_root / 'val2024'
    
    print(f"数据根目录: {data_root}")
    print(f"训练集标注文件: {train_ann}")
    print(f"验证集标注文件: {val_ann}")
    print(f"训练集图片目录: {train_img_dir}")
    print(f"验证集图片目录: {val_img_dir}")
    
    # 检查文件和目录是否存在
    if not train_ann.exists():
        raise FileNotFoundError(f"找不到训练集标注文件: {train_ann}")
    if not val_ann.exists():
        raise FileNotFoundError(f"找不到验证集标注文件: {val_ann}")
    if not train_img_dir.exists():
        raise FileNotFoundError(f"找不到训练集图片目录: {train_img_dir}")
    if not val_img_dir.exists():
        raise FileNotFoundError(f"找不到验证集图片目录: {val_img_dir}")
    
    # 加载数据
    with open(train_ann, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_ann, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
        
    # 获取当前最大ID
    max_img_id = max(img['id'] for img in train_data['images'])
    max_ann_id = max(ann['id'] for ann in train_data['annotations'])
    
    # 更新验证集的ID
    id_map = {}
    for img in val_data['images']:
        old_id = img['id']
        new_id = max_img_id + 1
        id_map[old_id] = new_id
        img['id'] = new_id
        max_img_id = new_id
        
    # 更新验证集标注的ID
    for ann in val_data['annotations']:
        ann['image_id'] = id_map[ann['image_id']]
        ann['id'] = max_ann_id + 1
        max_ann_id += 1
        
    # 合并数据
    merged_data = {
        'images': train_data['images'] + val_data['images'],
        'annotations': train_data['annotations'] + val_data['annotations'],
        'categories': train_data['categories']
    }
    
    print(f"合并后总图像数: {len(merged_data['images'])}")
    print(f"合并后总标注数: {len(merged_data['annotations'])}")
    
    return merged_data, [train_img_dir, val_img_dir]

def split_dataset(merged_data: Dict,
                 src_img_dirs: List[str],
                 save_dir: str,
                 train_ratio: float,
                 val_ratio: float,
                 seed: int):
    """分割数据集
    
    Args:
        merged_data: 合并后的数据集
        src_img_dirs: 源图片目录列表
        save_dir: 保存目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
    """
    save_dir = Path(save_dir)
    random.seed(seed)
    
    # 获取所有图像
    images = merged_data['images'].copy()
    random.shuffle(images)
    
    # 计算数据集大小
    total_size = len(images)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    print(f"\n数据集分割:")
    print(f"总样本数: {total_size}")
    print(f"训练集大小: {train_size} ({train_ratio*100:.1f}%)")
    print(f"验证集大小: {val_size} ({val_ratio*100:.1f}%)")
    print(f"未标注集大小: {total_size - train_size - val_size} ({(1-train_ratio-val_ratio)*100:.1f}%)")
    
    # 分割图像
    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    unlabeled_images = images[train_size + val_size:]
    
    # 获取图像ID集合
    train_ids = {img['id'] for img in train_images}
    val_ids = {img['id'] for img in val_images}
    unlabeled_ids = {img['id'] for img in unlabeled_images}  # 添加未标注图像ID集合
  
    # 分配标注
    train_anns = [
        ann for ann in merged_data['annotations']
        if ann['image_id'] in train_ids
    ]
    val_anns = [
        ann for ann in merged_data['annotations']
        if ann['image_id'] in val_ids
    ]
    # 保存未标注数据的标注
    unlabeled_anns = [
        ann for ann in merged_data['annotations']
        if ann['image_id'] in unlabeled_ids
    ]
    
    # 创建数据集字典
    datasets = {
        'labeled_train': {
            'images': train_images,
            'annotations': train_anns,
            'categories': merged_data['categories']
        },
        'labeled_val': {
            'images': val_images,
            'annotations': val_anns,
            'categories': merged_data['categories']
        },
        'unlabeled': {
            'images': unlabeled_images,
            'annotations': unlabeled_anns,  # 保留原始标注
            'categories': merged_data['categories']
        }
    }
    
    # 创建目录结构
    for name in ['labeled_train', 'labeled_val', 'unlabeled']:
        print(f"\n处理 {name} 数据集...")
        
        # 创建图片目录
        img_dir = save_dir / f'images_{name}'
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # 从所有源目录复制图片
        print(f"从以下目录复制图片:")
        for src_dir in src_img_dirs:
            print(f"  - {src_dir}")
        copy_images(datasets[name]['images'], src_img_dirs, img_dir)
            
        # 保存标注文件
        ann_dir = save_dir / 'annotations'
        ann_dir.mkdir(parents=True, exist_ok=True)
        ann_file = ann_dir / f'instances_{name}.json'
        
        with open(ann_file, 'w', encoding='utf-8') as f:
            json.dump(datasets[name], f, indent=2, ensure_ascii=False)
        print(f"保存标注到: {ann_file}")
        print(f"图像数量: {len(datasets[name]['images'])}")
        print(f"标注数量: {len(datasets[name]['annotations'])}")

def prepare_active_dataset(
    data_root: str,
    save_dir: str,
    train_ratio: float = 0.08,
    val_ratio: float = 0.02,
    seed: int = 42
):
    """准备主动学习数据集
    
    Args:
        data_root: 数据根目录
        save_dir: 保存目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
    """
    # 合并数据集
    merged_data, src_img_dirs = merge_coco_datasets(data_root)
    
    # 分割数据集
    split_dataset(merged_data, src_img_dirs, save_dir, train_ratio, val_ratio, seed)
    
    print("\n数据集准备完成!")
    print("\n目录结构:")
    print(f"{save_dir}/")
    print("├── images_labeled_train/")
    print("├── images_labeled_val/")
    print("├── images_unlabeled/")
    print("└── annotations/")
    print("    ├── instances_labeled_train.json")
    print("    ├── instances_labeled_val.json")
    print("    └── instances_unlabeled.json")

def parse_args():
    parser = argparse.ArgumentParser(description='准备主动学习数据集')
    parser.add_argument('data_root', help='数据根目录')
    parser.add_argument('save_dir', help='保存目录')
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.08,
        help='训练集比例 (默认: 0.08)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.02,
        help='验证集比例 (默认: 0.02)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    prepare_active_dataset(
        args.data_root,
        args.save_dir,
        args.train_ratio,
        args.val_ratio,
        args.seed
    )