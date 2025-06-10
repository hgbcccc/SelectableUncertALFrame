# import os
# import json
# from pathlib import Path

# def check_coco_structure(data_root):
#     """检查COCO数据集的目录结构"""
#     img_train_dir = Path(data_root) / 'train2017'
#     img_val_dir = Path(data_root) / 'val2017'
#     ann_file = Path(data_root) / 'annotations' / 'instances_train2017.json'  # 根据需要修改路径

#     if not img_train_dir.exists() or not img_train_dir.is_dir():
#         raise FileNotFoundError(f"训练图片目录不存在: {img_train_dir}")

#     if not img_val_dir.exists() or not img_val_dir.is_dir():
#         raise FileNotFoundError(f"验证图片目录不存在: {img_val_dir}")

#     if not ann_file.exists():
#         raise FileNotFoundError(f"标注文件不存在: {ann_file}")

#     print("COCO数据集的基本结构检查通过。")

# def validate_annotations(ann_file):
#     """验证标注文件的内容"""
#     with open(ann_file, 'r') as f:
#         annotations = json.load(f)

#     required_keys = ['images', 'annotations', 'categories']
#     for key in required_keys:
#         if key not in annotations:
#             raise ValueError(f"标注文件缺少必要的键: {key}")

#     print("标注文件的基本结构检查通过。")

#     # 检查每个图像的ID和文件名
#     image_ids = {img['id']: img['file_name'] for img in annotations['images']}
#     for ann in annotations['annotations']:
#         img_id = ann['image_id']
#         if img_id not in image_ids:
#             raise ValueError(f"标注中引用了不存在的图像ID: {img_id}")

#     print("所有标注的图像ID都有效。")

# def check_images(data_root):
#     """检查所有图像文件是否存在"""
#     img_train_dir = Path(data_root) / 'train2017'
#     img_val_dir = Path(data_root) / 'val2017'

#     # 检查训练集图像
#     for img_file in img_train_dir.glob('*'):
#         if not img_file.exists():
#             raise FileNotFoundError(f"训练图像文件不存在: {img_file}")

#     # 检查验证集图像
#     for img_file in img_val_dir.glob('*'):
#         if not img_file.exists():
#             raise FileNotFoundError(f"验证图像文件不存在: {img_file}")

#     print("所有图像文件都存在。")

# def validate_coco_dataset(data_root):
#     """验证COCO数据集的完整性"""
#     print("开始验证COCO数据集...")
#     try:
#         check_coco_structure(data_root)
#         ann_file = Path(data_root) / 'annotations' / 'instances_train2017.json'  # 根据需要修改路径
#         validate_annotations(ann_file)
#         check_images(data_root)
#         print("\n✅ 验证通过: 所有文件都是有效的")
#     except Exception as e:
#         print(f"\n❌ 验证失败: {str(e)}")
#         print("请检查上述错误")

# if __name__ == "__main__":
#     # 设置数据集根目录
#     data_root = 'data/coco'  # 根据实际路径修改
#     validate_coco_dataset(data_root)

import os
import json
from pathlib import Path

def check_coco_structure(data_root):
    """检查COCO数据集的目录结构"""
    img_train_dir = Path(data_root) / 'train2017'
    img_val_dir = Path(data_root) / 'val2017'
    ann_file = Path(data_root) / 'annotations' / 'instances_train2017.json'  # 根据需要修改路径

    if not img_train_dir.exists() or not img_train_dir.is_dir():
        raise FileNotFoundError(f"训练图片目录不存在: {img_train_dir}")

    if not img_val_dir.exists() or not img_val_dir.is_dir():
        raise FileNotFoundError(f"验证图片目录不存在: {img_val_dir}")

    if not ann_file.exists():
        raise FileNotFoundError(f"标注文件不存在: {ann_file}")

    print(f"{data_root} 的基本结构检查通过。")

def validate_annotations(ann_file):
    """验证标注文件的内容"""
    with open(ann_file, 'r') as f:
        annotations = json.load(f)

    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in annotations:
            raise ValueError(f"标注文件缺少必要的键: {key}")

    print("标注文件的基本结构检查通过。")

    # 检查每个图像的ID和文件名
    image_ids = {img['id']: img['file_name'] for img in annotations['images']}
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        if img_id not in image_ids:
            raise ValueError(f"标注中引用了不存在的图像ID: {img_id}")

    print("所有标注的图像ID都有效。")

def check_images(data_root):
    """检查所有图像文件是否存在"""
    img_train_dir = Path(data_root) / 'train2017'
    img_val_dir = Path(data_root) / 'val2017'

    # 检查训练集图像
    for img_file in img_train_dir.glob('*'):
        if not img_file.exists():
            raise FileNotFoundError(f"训练图像文件不存在: {img_file}")

    # 检查验证集图像
    for img_file in img_val_dir.glob('*'):
        if not img_file.exists():
            raise FileNotFoundError(f"验证图像文件不存在: {img_file}")

    print("所有图像文件都存在。")

def validate_coco_dataset(data_root):
    """验证COCO数据集的完整性"""
    print(f"开始验证数据集: {data_root}...")
    try:
        check_coco_structure(data_root)
        ann_file = Path(data_root) / 'annotations' / 'instances_train2017.json'  # 根据需要修改路径
        validate_annotations(ann_file)
        check_images(data_root)
        print("\n✅ 验证通过: 所有文件都是有效的")
    except Exception as e:
        print(f"\n❌ 验证失败: {str(e)}")
        print("请检查上述错误")

def validate_all_coco_datasets(base_dir):
    """遍历base_dir下的所有COCO数据集并进行验证"""
    for dataset_dir in Path(base_dir).iterdir():
        if dataset_dir.is_dir():
            validate_coco_dataset(dataset_dir)

if __name__ == "__main__":
    # 设置数据集根目录
    base_dir = 'data'  # 根据实际路径修改
    validate_all_coco_datasets(base_dir)