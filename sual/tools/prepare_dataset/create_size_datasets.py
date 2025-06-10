import json
import os
import shutil
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
'''这个脚本用于创建不同大小的数据集，划分的依据是边界框的面积，分成标注框面积小，中，大三个数据集 
    使用方法：
    python create_size_datasets.py
    目录结构：
        data/ForestDamages/
            ├── annotations/
            │   ├── instances_train2024.json
            │   └── instances_val2024.json
            ├── train2024/
            │   └── [训练集图片]
            └── val2024/
                └── [验证集图片]

    输出目录：
        data/ForestDamages/
            └── small_datasets/
                ├── small_objects/
                │   ├── images/
                │   └── annotations/
                ├── medium_objects/
                └── large_objects/  
'''
def merge_json_files(train_json, val_json):
    """合并训练集和验证集的JSON文件"""
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    with open(val_json, 'r') as f:
        val_data = json.load(f)
    
    # 合并images和annotations
    merged_data = {
        'info': train_data['info'],
        'licenses': train_data['licenses'],
        'categories': train_data['categories'],
        'images': train_data['images'] + val_data['images'],
        'annotations': train_data['annotations'] + val_data['annotations']
    }
    
    return merged_data

def calculate_area_distribution(annotations):
    """计算所有边界框的面积分布"""
    areas = []
    for ann in annotations:
        areas.append(ann['area'])
    
    # 计算1/3和2/3分位点
    area_thresholds = np.percentile(areas, [33.33, 66.67])
    
    # 绘制面积分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(areas, bins=50)
    plt.axvline(area_thresholds[0], color='r', linestyle='--', label='1/3分位点')
    plt.axvline(area_thresholds[1], color='g', linestyle='--', label='2/3分位点')
    plt.xlabel('边界框面积')
    plt.ylabel('数量')
    plt.title('边界框面积分布')
    plt.legend()
    plt.savefig('area_distribution.png')
    plt.close()
    
    return area_thresholds

def classify_images_by_box_size(merged_data, area_thresholds):
    """根据边界框大小对图片进行分类"""
    image_areas = defaultdict(list)  # 存储每张图片中所有框的面积
    
    # 创建图片ID到图片信息的映射
    image_id_to_info = {img['id']: img for img in merged_data['images']}
    
    # 收集每张图片中的所有框面积
    for ann in merged_data['annotations']:
        image_areas[ann['image_id']].append(ann['area'])
    
    # 根据框面积对图片进行分类
    small_images = []
    medium_images = []
    large_images = []
    
    for image_id, areas in image_areas.items():
        mean_area = np.mean(areas)  # 使用平均面积作为该图片的特征
        image_info = image_id_to_info[image_id]
        
        if mean_area <= area_thresholds[0]:
            small_images.append(image_info)
        elif mean_area <= area_thresholds[1]:
            medium_images.append(image_info)
        else:
            large_images.append(image_info)
    
    return small_images, medium_images, large_images

def copy_images_and_create_annotations(image_lists, source_dirs, target_base_dir):
    """复制图片到目标目录并创建对应的标注文件"""
    size_names = ['small', 'medium', 'large']
    
    # 确保目标目录存在
    os.makedirs(target_base_dir, exist_ok=True)
    
    for size_name, images in zip(size_names, image_lists):
        # 创建目标目录
        target_dir = os.path.join(target_base_dir, f'{size_name}_objects')
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'annotations'), exist_ok=True)
        
        # 复制图片
        print(f'Copying {size_name} object images...')
        for img in tqdm(images):
            file_name = img['file_name']
            # 在source_dirs中查找图片
            for source_dir in source_dirs:
                source_path = os.path.join(source_dir, file_name)
                if os.path.exists(source_path):
                    target_path = os.path.join(target_dir, 'images', file_name)
                    shutil.copy2(source_path, target_path)
                    break
        
        # 创建对应的标注文件
        ann_file = {
            'images': images,
            'categories': merged_data['categories'],
            'annotations': [ann for ann in merged_data['annotations'] 
                          if ann['image_id'] in [img['id'] for img in images]]
        }
        
        with open(os.path.join(target_dir, 'annotations', 'instances.json'), 'w') as f:
            json.dump(ann_file, f)

if __name__ == '__main__':
    # 设置路径
    base_dir = 'data/ForestDamages'
    train_json = os.path.join(base_dir, 'annotations/instances_train2024.json')
    val_json = os.path.join(base_dir, 'annotations/instances_val2024.json')
    source_dirs = [
        os.path.join(base_dir, 'train2024'),
        os.path.join(base_dir, 'val2024')
    ]
    target_base_dir = os.path.join(base_dir, 'small_datasets')
    
    # 1. 合并JSON文件
    print('Merging JSON files...')
    merged_data = merge_json_files(train_json, val_json)
    
    # 2. 计算面积分布
    print('Calculating area distribution...')
    area_thresholds = calculate_area_distribution(merged_data['annotations'])
    print(f'Area thresholds: {area_thresholds}')
    
    # 3. 分类图片
    print('Classifying images...')
    small_images, medium_images, large_images = classify_images_by_box_size(
        merged_data, area_thresholds)
    
    print(f'Number of images with small objects: {len(small_images)}')
    print(f'Number of images with medium objects: {len(medium_images)}')
    print(f'Number of images with large objects: {len(large_images)}')
    
    # 4. 复制图片和创建标注文件
    copy_images_and_create_annotations(
        [small_images, medium_images, large_images],
        source_dirs,
        target_base_dir
    )
    
    print('Done!')