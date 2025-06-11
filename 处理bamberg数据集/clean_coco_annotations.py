import os
import json
import shutil
from typing import Dict, List

def clean_coco_annotations(image_dir: str, annotation_file: str, output_file: str) -> Dict:
    """
    清理COCO标注文件中没有对应图片的标注项
    
    参数:
        image_dir: 图片文件夹路径
        annotation_file: 原始COCO标注JSON文件路径
        output_file: 清理后的输出JSON文件路径
    
    返回:
        Tuple[Dict, int]: 返回清理后的COCO标注数据和原始图片数量
    """
    # 读取原始标注文件
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # 获取文件夹中的所有图片文件
    actual_images = set()
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            actual_images.add(filename)
    
    # 筛选有对应图片的标注
    valid_images = [img for img in coco_data['images'] if img['file_name'] in actual_images]
    valid_image_ids = {img['id'] for img in valid_images}
    
    # 筛选对应的标注
    valid_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in valid_image_ids]
    
    # 构建新的COCO数据
    cleaned_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data.get('categories', []),
        'images': valid_images,
        'annotations': valid_annotations
    }
    
    # 保存清理后的文件
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    return cleaned_data, len(coco_data['images'])

if __name__ == '__main__':
    # 直接在这里设置路径
    IMAGE_DIR = "data/Bamberg_coco1024/val2024_jpg"  # 图片文件夹路径
    ANNOTATION_FILE = "data/Bamberg_coco1024/annotations/instances_val2024_jpg.json"  # 原始标注文件路径
    OUTPUT_FILE = "data/Bamberg_coco1024/annotations/instances_val2024_jpg_cleaned.json"  # 输出文件路径
    CREATE_BACKUP = False  # 是否创建备份
    
    # 如果需要备份原始文件
    if CREATE_BACKUP:
        backup_file = ANNOTATION_FILE + '.bak'
        shutil.copy2(ANNOTATION_FILE, backup_file)
        print(f"已创建原始文件的备份: {backup_file}")
    
    # 执行清理并获取原始图片数量
    cleaned_data, original_count = clean_coco_annotations(IMAGE_DIR, ANNOTATION_FILE, OUTPUT_FILE)
    
    # 输出统计信息
    cleaned_count = len(cleaned_data['images'])
    removed_count = original_count - cleaned_count
    
    print(f"\n清理完成!")
    print(f"原始标注图片数量: {original_count}")
    print(f"清理后标注图片数量: {cleaned_count}")
    print(f"移除的无效标注数量: {removed_count}")
    print(f"清理后的文件已保存到: {OUTPUT_FILE}")