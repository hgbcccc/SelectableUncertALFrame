import os
import glob
import json
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import datetime
import shutil

def xml_to_coco(dataset_path, output_path, categories=None):
    """
    将XML标注转换为COCO格式
    
    Args:
        dataset_path: 数据集根目录路径
        output_path: 输出COCO文件路径
        categories: 预定义类别列表，如果为None则自动从数据中提取
    """
    # 初始化COCO数据结构
    coco_output = {
        "info": {
            "description": "Spruce Bark Beetle Dataset",
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 类别ID映射
    category_map = {}
    
    # 如果预定义了类别列表
    if categories:
        for i, cat in enumerate(categories):
            category_id = i + 1  # COCO类别ID从1开始
            coco_output["categories"].append({
                "id": category_id,
                "name": cat,
                "supercategory": "tree"
            })
            category_map[cat] = category_id
    
    # 全局计数器
    image_id = 0
    annotation_id = 0
    
    # 获取垂直和倾斜拍摄的子目录
    vertical_dirs = glob.glob(os.path.join(dataset_path, "vertical", "*"))
    oblique_dirs = glob.glob(os.path.join(dataset_path, "oblique", "*"))
    
    vertical_dirs = [d for d in vertical_dirs if os.path.isdir(d)]
    oblique_dirs = [d for d in oblique_dirs if os.path.isdir(d)]
    
    all_dirs = vertical_dirs + oblique_dirs
    
    # 输出一些调试信息
    print(f"找到 {len(vertical_dirs)} 个垂直拍摄目录和 {len(oblique_dirs)} 个倾斜拍摄目录")
    
    # 第一遍：收集所有类别（如果需要）
    if not categories:
        print("第一遍：收集类别信息...")
        unique_categories = set()
        
        for directory in tqdm(all_dirs, desc="收集类别"):
            annotations_dir = os.path.join(directory, "Annotations")
            if not os.path.exists(annotations_dir):
                continue
                
            xml_files = glob.glob(os.path.join(annotations_dir, "*.xml"))
            for xml_file in xml_files:
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    
                    for obj in root.findall('object'):
                        # 使用tree标签代替name标签
                        tree_elem = obj.find('tree')
                        if tree_elem is not None and tree_elem.text is not None:
                            class_name = tree_elem.text.strip()
                            if class_name:
                                unique_categories.add(class_name)
                except Exception as e:
                    print(f"收集类别时出错 {xml_file}: {str(e)}")
                    continue
        
        # 创建类别列表和映射
        for i, cat in enumerate(sorted(unique_categories)):
            category_id = i + 1
            coco_output["categories"].append({
                "id": category_id,
                "name": cat,
                "supercategory": "tree"
            })
            category_map[cat] = category_id
            
        print(f"找到 {len(unique_categories)} 个类别: {', '.join(sorted(unique_categories))}")
    
    # 第二遍：处理所有图片和标注
    print("\n第二遍：转换数据...")
    
    # 统计计数器
    total_image_count = 0
    total_xml_count = 0
    total_annotation_count = 0
    xml_parsing_errors = 0
    
    for directory in tqdm(all_dirs, desc="处理目录"):
        dir_type = "vertical" if "/vertical/" in directory.replace("\\", "/") else "oblique"
        dir_name = os.path.basename(directory)
        
        images_dir = os.path.join(directory, "Images")
        annotations_dir = os.path.join(directory, "Annotations")
        
        if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
            continue
            
        # 获取所有图片
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
        
        total_image_count += len(image_files)
        xml_files = glob.glob(os.path.join(annotations_dir, "*.xml"))
        total_xml_count += len(xml_files)
        
        # 处理每个图片及其标注
        for img_file in image_files:
            basename = os.path.splitext(os.path.basename(img_file))[0]
            xml_file = os.path.join(annotations_dir, f"{basename}.xml")
            
            if not os.path.exists(xml_file):
                continue
                
            try:
                # 获取图片信息
                img = Image.open(img_file)
                width, height = img.size
                
                # 添加图片信息
                image_id += 1
                
                # 简化文件名，只使用基本名称
                file_name = os.path.basename(img_file)
                
                coco_output["images"].append({
                    "id": image_id,
                    "license": 1,
                    "file_name": file_name,
                    "height": height,
                    "width": width,
                    "date_captured": ""
                })
                
                # 解析XML标注
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                obj_count = 0
                for obj in root.findall('object'):
                    obj_count += 1
                    
                    # 获取类别 - 使用tree标签代替name标签
                    tree_elem = obj.find('tree')
                    if tree_elem is None or tree_elem.text is None:
                        continue
                        
                    class_name = tree_elem.text.strip()
                    if not class_name:
                        continue
                        
                    # 检查类别是否在预定义列表中
                    if class_name not in category_map:
                        if categories is not None:
                            print(f"警告：发现未知类别 '{class_name}'，跳过")
                            continue
                        else:
                            category_id = len(category_map) + 1
                            coco_output["categories"].append({
                                "id": category_id,
                                "name": class_name,
                                "supercategory": "tree"
                            })
                            category_map[class_name] = category_id
                    
                    category_id = category_map[class_name]
                    
                    # 检查是否有损伤信息
                    damage_elem = obj.find('damage')
                    damage_type = damage_elem.text.strip() if damage_elem is not None and damage_elem.text is not None else "unknown"
                    
                    # 获取边界框
                    bndbox = obj.find('bndbox')
                    if bndbox is None:
                        continue
                        
                    try:
                        xmin = float(bndbox.find('xmin').text)
                        ymin = float(bndbox.find('ymin').text)
                        xmax = float(bndbox.find('xmax').text)
                        ymax = float(bndbox.find('ymax').text)
                        
                        # 检查坐标有效性
                        if xmin >= xmax or ymin >= ymax:
                            continue
                            
                        # COCO格式：[x, y, width, height]
                        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                        area = bbox[2] * bbox[3]
                        
                        # 添加标注
                        annotation_id += 1
                        coco_output["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": bbox,
                            "area": area,
                            "segmentation": [],
                            "iscrowd": 0,
                            "damage_type": damage_type  # 添加损伤类型作为额外属性
                        })
                        total_annotation_count += 1
                        
                    except (ValueError, AttributeError) as e:
                        print(f"处理边界框出错 ({xml_file}): {str(e)}")
                        continue
                
                if obj_count == 0:
                    print(f"警告: XML文件 {xml_file} 中没有找到任何目标对象")
                        
            except Exception as e:
                xml_parsing_errors += 1
                print(f"处理文件出错 {xml_file}: {str(e)}")
    
    # 保存COCO JSON文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, indent=4)
    
    print(f"\n转换统计:")
    print(f"- 总图片数: {total_image_count}")
    print(f"- 总XML文件数: {total_xml_count}")
    print(f"- 转换后的图片数: {len(coco_output['images'])}")
    print(f"- 转换后的标注数: {len(coco_output['annotations'])}")
    print(f"- XML解析错误: {xml_parsing_errors}")
    print(f"\nCOCO数据保存至: {output_path}")
    
    return coco_output

def create_standard_coco_dataset(coco_data, dataset_path, output_dir, train_ratio=0.9, random_seed=42):
    """创建标准COCO数据集格式"""
    import random
    random.seed(random_seed)
    
    # 创建标准目录结构
    annotations_dir = os.path.join(output_dir, "annotations")
    train_images_dir = os.path.join(output_dir, "train2024")
    val_images_dir = os.path.join(output_dir, "val2024")
    
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    
    # 获取所有图片ID
    all_image_ids = [img["id"] for img in coco_data["images"]]
    random.shuffle(all_image_ids)
    
    # 拆分图片ID
    train_count = int(len(all_image_ids) * train_ratio)
    train_image_ids = set(all_image_ids[:train_count])
    val_image_ids = set(all_image_ids[train_count:])
    
    # 创建训练集和验证集
    train_data = {
        "info": coco_data["info"],
        "licenses": coco_data["licenses"],
        "categories": coco_data["categories"],
        "images": [],
        "annotations": []
    }
    
    val_data = {
        "info": coco_data["info"],
        "licenses": coco_data["licenses"],
        "categories": coco_data["categories"],
        "images": [],
        "annotations": []
    }
    
    # 分配图片
    for img in coco_data["images"]:
        img_id = img["id"]
        if img_id in train_image_ids:
            train_data["images"].append(img)
        else:
            val_data["images"].append(img)
    
    # 分配标注
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id in train_image_ids:
            train_data["annotations"].append(ann)
        else:
            val_data["annotations"].append(ann)
    
    # 保存注释文件
    train_json = os.path.join(annotations_dir, "instances_train2024.json")
    val_json = os.path.join(annotations_dir, "instances_val2024.json")
    
    with open(train_json, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)
    
    with open(val_json, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=4)
    
    # 复制图片文件
    print("\n复制图片文件...")
    
    # 查找所有图片文件
    all_image_files = []
    vertical_dirs = glob.glob(os.path.join(dataset_path, "vertical", "*", "Images"))
    oblique_dirs = glob.glob(os.path.join(dataset_path, "oblique", "*", "Images"))
    
    for image_dir in vertical_dirs + oblique_dirs:
        for ext in ['.jpg', '.jpeg', '.png']:
            all_image_files.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
    
    # 创建文件名到路径的映射
    file_to_path = {}
    for img_path in all_image_files:
        file_to_path[os.path.basename(img_path)] = img_path
    
    # 复制训练集图片
    for img in tqdm(train_data["images"], desc="复制训练集图片"):
        file_name = img["file_name"]
        if file_name in file_to_path:
            src_path = file_to_path[file_name]
            dst_path = os.path.join(train_images_dir, file_name)
            shutil.copy2(src_path, dst_path)
    
    # 复制验证集图片
    for img in tqdm(val_data["images"], desc="复制验证集图片"):
        file_name = img["file_name"]
        if file_name in file_to_path:
            src_path = file_to_path[file_name]
            dst_path = os.path.join(val_images_dir, file_name)
            shutil.copy2(src_path, dst_path)
    
    print(f"\n数据集创建完成！")
    print(f"训练集: {len(train_data['images'])} 张图片, {len(train_data['annotations'])} 个标注")
    print(f"验证集: {len(val_data['images'])} 张图片, {len(val_data['annotations'])} 个标注")
    
    return train_json, val_json

def main():
    # 指定数据集路径
    dataset_path = "../data/Data_Set_Spruce_Bark_Beetle"
    
    # 创建输出目录
    output_dir = "../data/coco_spruce_beetle"
    temp_dir = "../data/temp_coco"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 预定义类别（根据您的数据集可能需要修改）
    categories = ["Spruce", "Pine", "Birch", "Aspen", "Other"]
    
    # 转换为COCO格式
    temp_file = os.path.join(temp_dir, "annotations.json")
    coco_data = xml_to_coco(dataset_path, temp_file, categories)
    
    # 创建标准COCO数据集
    train_json, val_json = create_standard_coco_dataset(coco_data, dataset_path, output_dir)
    
    print("\n=== 处理完成 ===")
    print(f"标准COCO数据集保存在: {output_dir}")
    print(f"训练集标注文件: {train_json}")
    print(f"验证集标注文件: {val_json}")

if __name__ == "__main__":
    main()