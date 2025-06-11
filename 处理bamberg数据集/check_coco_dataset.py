#!/usr/bin/env python3
import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm

def validate_coco_dataset(
    data_root='data/Bamberg_coco1024',
    train_anno_file='annotations/instances_train2024.json',
    val_anno_file='annotations/instances_val2024.json',
    train_img_dir='train2024',
    val_img_dir='val2024',
    save_report=True,
    report_dir='data/Bamberg_coco1024/validation_report'
):
    """验证COCO数据集图片和标注的匹配情况并保存问题报告"""
    # 构建完整路径
    train_anno_path = os.path.join(data_root, train_anno_file)
    val_anno_path = os.path.join(data_root, val_anno_file)
    train_img_path = os.path.join(data_root, train_img_dir)
    val_img_path = os.path.join(data_root, val_img_dir)
    
    # 创建报告目录
    if save_report and not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    print(f"检查训练集标注文件: {train_anno_path}")
    print(f"检查验证集标注文件: {val_anno_path}")
    print(f"检查训练集图片目录: {train_img_path}")
    print(f"检查验证集图片目录: {val_img_path}")
    
    # 检查文件/目录存在
    missing_files = []
    for path, name in [
        (train_anno_path, "训练集标注文件"),
        (val_anno_path, "验证集标注文件"),
        (train_img_path, "训练集图片目录"),
        (val_img_path, "验证集图片目录")
    ]:
        if not os.path.exists(path):
            print(f"错误: {name}不存在: {path}")
            missing_files.append(path)
    
    if missing_files:
        return False
    
    # 获取磁盘上的图片文件列表
    print("\n统计磁盘中的图片文件...")
    train_images_on_disk = set(os.listdir(train_img_path))
    val_images_on_disk = set(os.listdir(val_img_path))
    
    # 过滤非图片文件
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    train_images_on_disk = {f for f in train_images_on_disk 
                            if os.path.splitext(f.lower())[1] in valid_extensions}
    val_images_on_disk = {f for f in val_images_on_disk 
                          if os.path.splitext(f.lower())[1] in valid_extensions}
    
    # 创建文件名映射 (不含扩展名 -> 全文件名)，用于处理扩展名不匹配的情况
    train_filename_map = {}
    for f in train_images_on_disk:
        name_without_ext = os.path.splitext(f)[0]
        train_filename_map[name_without_ext] = f
    
    val_filename_map = {}
    for f in val_images_on_disk:
        name_without_ext = os.path.splitext(f)[0]
        val_filename_map[name_without_ext] = f
    
    print(f"在磁盘上找到:")
    print(f"- 训练集图片: {len(train_images_on_disk)} 个")
    print(f"- 验证集图片: {len(val_images_on_disk)} 个")
    
    # 加载标注文件
    print("\n读取标注文件...")
    try:
        with open(train_anno_path, 'r', encoding='utf-8') as f:
            train_anno = json.load(f)
    except Exception as e:
        print(f"错误: 无法加载训练集标注文件: {e}")
        return False
    
    try:
        with open(val_anno_path, 'r', encoding='utf-8') as f:
            val_anno = json.load(f)
    except Exception as e:
        print(f"错误: 无法加载验证集标注文件: {e}")
        return False
    
    # 检查COCO格式
    for dataset_name, dataset in [("训练集", train_anno), ("验证集", val_anno)]:
        for field in ['images', 'annotations', 'categories']:
            if field not in dataset:
                print(f"错误: {dataset_name}标注缺少必要字段 '{field}'")
                return False
    
    # 提取标注文件中的图片信息
    train_images_in_anno = {}  # {filename: image_id}
    train_img_id_to_filename = {}  # {image_id: filename}
    for img in train_anno['images']:
        file_name = img.get('file_name', '')
        # 处理文件名中可能包含的路径
        base_name = os.path.basename(file_name)
        train_images_in_anno[base_name] = img['id']
        train_img_id_to_filename[img['id']] = base_name
    
    val_images_in_anno = {}
    val_img_id_to_filename = {}
    for img in val_anno['images']:
        file_name = img.get('file_name', '')
        base_name = os.path.basename(file_name)
        val_images_in_anno[base_name] = img['id']
        val_img_id_to_filename[img['id']] = base_name
    
    print(f"\n在标注文件中发现:")
    print(f"- 训练集图片条目: {len(train_images_in_anno)} 个")
    print(f"- 验证集图片条目: {len(val_images_in_anno)} 个")
    
    # 检查图片文件名中的路径前缀
    train_has_prefix = any('/' in img['file_name'] for img in train_anno['images'])
    val_has_prefix = any('/' in img['file_name'] for img in val_anno['images'])
    
    print(f"\n文件路径分析:")
    print(f"- 训练集标注中的文件名{'包含' if train_has_prefix else '不包含'}路径前缀")
    print(f"- 验证集标注中的文件名{'包含' if val_has_prefix else '不包含'}路径前缀")
    
    # 统计每个图片的标注数量
    train_anno_counts = defaultdict(int)
    for anno in train_anno['annotations']:
        img_id = anno.get('image_id')
        if img_id in train_img_id_to_filename:
            filename = train_img_id_to_filename[img_id]
            train_anno_counts[filename] += 1
    
    val_anno_counts = defaultdict(int)
    for anno in val_anno['annotations']:
        img_id = anno.get('image_id')
        if img_id in val_img_id_to_filename:
            filename = val_img_id_to_filename[img_id]
            val_anno_counts[filename] += 1
    
    # 一致性检查
    print("\n执行一致性检查...")
    
    # 1. 在标注中但不在磁盘上的图片
    train_missing_images = set()
    train_ext_mismatch = {}  # 扩展名不匹配的文件
    
    for filename in train_images_in_anno:
        if filename not in train_images_on_disk:
            # 检查是否是扩展名问题
            name_without_ext = os.path.splitext(filename)[0]
            if name_without_ext in train_filename_map:
                train_ext_mismatch[filename] = train_filename_map[name_without_ext]
            else:
                train_missing_images.add(filename)
    
    val_missing_images = set()
    val_ext_mismatch = {}
    
    for filename in val_images_in_anno:
        if filename not in val_images_on_disk:
            # 检查是否是扩展名问题
            name_without_ext = os.path.splitext(filename)[0]
            if name_without_ext in val_filename_map:
                val_ext_mismatch[filename] = val_filename_map[name_without_ext]
            else:
                val_missing_images.add(filename)
    
    # 2. 在磁盘上但不在标注中的图片
    train_unlabeled_images = train_images_on_disk - {f for f in train_images_in_anno if f in train_images_on_disk}
    # 排除扩展名不匹配但已有对应的图片
    train_unlabeled_images = {f for f in train_unlabeled_images 
                            if os.path.splitext(f)[0] not in {os.path.splitext(k)[0] for k in train_ext_mismatch}}
    
    val_unlabeled_images = val_images_on_disk - {f for f in val_images_in_anno if f in val_images_on_disk}
    val_unlabeled_images = {f for f in val_unlabeled_images 
                          if os.path.splitext(f)[0] not in {os.path.splitext(k)[0] for k in val_ext_mismatch}}
    
    # 3. 没有标注的图片（在标注文件中但没有任何标注）
    train_zero_anno_images = {img for img in train_images_in_anno 
                           if train_anno_counts[img] == 0}
    val_zero_anno_images = {img for img in val_images_in_anno 
                         if val_anno_counts[img] == 0}
    
    # 打印检查结果
    print("\n=== 训练集检查结果 ===")
    print(f"在标注中但不在磁盘上的图片: {len(train_missing_images)}")
    if train_missing_images:
        print("示例:")
        for img in list(train_missing_images)[:5]:
            print(f"  - {img}")
        if len(train_missing_images) > 5:
            print(f"  - ... 还有 {len(train_missing_images) - 5} 个")
    
    print(f"\n文件扩展名不匹配的图片: {len(train_ext_mismatch)}")
    if train_ext_mismatch:
        print("示例:")
        for i, (old, new) in enumerate(list(train_ext_mismatch.items())[:5]):
            print(f"  - {old} → {new}")
        if len(train_ext_mismatch) > 5:
            print(f"  - ... 还有 {len(train_ext_mismatch) - 5} 个")
    
    print(f"\n在磁盘上但不在标注中的图片: {len(train_unlabeled_images)}")
    if train_unlabeled_images:
        print("示例:")
        for img in list(train_unlabeled_images)[:5]:
            print(f"  - {img}")
        if len(train_unlabeled_images) > 5:
            print(f"  - ... 还有 {len(train_unlabeled_images) - 5} 个")
    
    print(f"\n在标注中但没有任何标注的图片: {len(train_zero_anno_images)}")
    if train_zero_anno_images:
        print("示例:")
        for img in list(train_zero_anno_images)[:5]:
            print(f"  - {img}")
        if len(train_zero_anno_images) > 5:
            print(f"  - ... 还有 {len(train_zero_anno_images) - 5} 个")
    
    # 验证集结果
    print("\n=== 验证集检查结果 ===")
    print(f"在标注中但不在磁盘上的图片: {len(val_missing_images)}")
    if val_missing_images:
        print("示例:")
        for img in list(val_missing_images)[:5]:
            print(f"  - {img}")
        if len(val_missing_images) > 5:
            print(f"  - ... 还有 {len(val_missing_images) - 5} 个")
    
    print(f"\n文件扩展名不匹配的图片: {len(val_ext_mismatch)}")
    if val_ext_mismatch:
        print("示例:")
        for i, (old, new) in enumerate(list(val_ext_mismatch.items())[:5]):
            print(f"  - {old} → {new}")
        if len(val_ext_mismatch) > 5:
            print(f"  - ... 还有 {len(val_ext_mismatch) - 5} 个")
    
    print(f"\n在磁盘上但不在标注中的图片: {len(val_unlabeled_images)}")
    if val_unlabeled_images:
        print("示例:")
        for img in list(val_unlabeled_images)[:5]:
            print(f"  - {img}")
        if len(val_unlabeled_images) > 5:
            print(f"  - ... 还有 {len(val_unlabeled_images) - 5} 个")
    
    print(f"\n在标注中但没有任何标注的图片: {len(val_zero_anno_images)}")
    if val_zero_anno_images:
        print("示例:")
        for img in list(val_zero_anno_images)[:5]:
            print(f"  - {img}")
        if len(val_zero_anno_images) > 5:
            print(f"  - ... 还有 {len(val_zero_anno_images) - 5} 个")
    
    # 检查类别信息
    print("\n=== 类别信息检查 ===")
    train_categories = {cat['id']: cat['name'] for cat in train_anno['categories']}
    val_categories = {cat['id']: cat['name'] for cat in val_anno['categories']}
    
    print(f"训练集类别数量: {len(train_categories)}")
    print(f"验证集类别数量: {len(val_categories)}")
    
    # 检查类别一致性
    if train_categories.keys() != val_categories.keys():
        print("警告: 训练集和验证集类别ID不一致")
        train_only = set(train_categories.keys()) - set(val_categories.keys())
        val_only = set(val_categories.keys()) - set(train_categories.keys())
        
        if train_only:
            print(f"训练集独有类别: {train_only}")
        if val_only:
            print(f"验证集独有类别: {val_only}")
    
    # 检查类别使用情况
    train_category_counts = defaultdict(int)
    for anno in train_anno['annotations']:
        if 'category_id' in anno:
            cat_id = anno['category_id']
            train_category_counts[cat_id] += 1
    
    val_category_counts = defaultdict(int)
    for anno in val_anno['annotations']:
        if 'category_id' in anno:
            cat_id = anno['category_id']
            val_category_counts[cat_id] += 1
    
    print("\n类别使用情况:")
    print("训练集:")
    for cat_id, name in sorted(train_categories.items()):
        count = train_category_counts[cat_id]
        print(f"  - {name} (ID: {cat_id}): {count} 个标注")
    
    print("\n验证集:")
    for cat_id, name in sorted(val_categories.items()):
        count = val_category_counts[cat_id]
        print(f"  - {name} (ID: {cat_id}): {count} 个标注")
    
    # 检查边界框
    print("\n=== 边界框检查 ===")
    
    def check_bboxes(annotations, dataset_name):
        invalid_bboxes = []
        negative_coords = []
        zero_width_height = []
        
        for i, anno in enumerate(tqdm(annotations, desc=f"检查{dataset_name}边界框")):
            if 'bbox' in anno:
                bbox = anno['bbox']
                
                # 检查边界框的基本格式
                if len(bbox) != 4:
                    invalid_bboxes.append(i)
                    continue
                
                # 检查负坐标
                if bbox[0] < 0 or bbox[1] < 0:
                    negative_coords.append(i)
                
                # 检查零宽高
                if bbox[2] <= 0 or bbox[3] <= 0:
                    zero_width_height.append(i)
        
        return {
            "invalid_format": invalid_bboxes,
            "negative_coords": negative_coords,
            "zero_width_height": zero_width_height
        }
    
    train_bbox_issues = check_bboxes(train_anno['annotations'], "训练集")
    val_bbox_issues = check_bboxes(val_anno['annotations'], "验证集")
    
    print(f"训练集边界框问题:")
    print(f"- 格式错误: {len(train_bbox_issues['invalid_format'])}")
    print(f"- 负坐标: {len(train_bbox_issues['negative_coords'])}")
    print(f"- 零宽/高: {len(train_bbox_issues['zero_width_height'])}")
    
    print(f"\n验证集边界框问题:")
    print(f"- 格式错误: {len(val_bbox_issues['invalid_format'])}")
    print(f"- 负坐标: {len(val_bbox_issues['negative_coords'])}")
    print(f"- 零宽/高: {len(val_bbox_issues['zero_width_height'])}")
    
    # 总结报告
    print("\n=== 总结 ===")
    
    has_issues = (
        train_missing_images or val_missing_images or
        train_ext_mismatch or val_ext_mismatch or
        train_unlabeled_images or val_unlabeled_images or
        train_zero_anno_images or val_zero_anno_images or
        any(len(issues) > 0 for issues in train_bbox_issues.values()) or
        any(len(issues) > 0 for issues in val_bbox_issues.values())
    )
    
    # 保存检查结果到文件
    if save_report:
        report = {
            "train": {
                "missing_images": list(train_missing_images),
                "ext_mismatch": train_ext_mismatch,
                "unlabeled_images": list(train_unlabeled_images),
                "zero_anno_images": list(train_zero_anno_images),
                "bbox_issues": train_bbox_issues
            },
            "val": {
                "missing_images": list(val_missing_images),
                "ext_mismatch": val_ext_mismatch,
                "unlabeled_images": list(val_unlabeled_images),
                "zero_anno_images": list(val_zero_anno_images),
                "bbox_issues": val_bbox_issues
            },
            "categories": {
                "train": train_categories,
                "val": val_categories
            },
            "path_info": {
                "train_has_prefix": train_has_prefix,
                "val_has_prefix": val_has_prefix
            }
        }
        
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "validation_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"详细报告已保存至: {report_path}")
        
        # 为每种问题单独保存列表文件，方便后续处理
        train_ext_mismatch_path = os.path.join(report_dir, "train_ext_mismatch.json")
        with open(train_ext_mismatch_path, 'w', encoding='utf-8') as f:
            json.dump(train_ext_mismatch, f, ensure_ascii=False, indent=2)
        
        val_ext_mismatch_path = os.path.join(report_dir, "val_ext_mismatch.json")
        with open(val_ext_mismatch_path, 'w', encoding='utf-8') as f:
            json.dump(val_ext_mismatch, f, ensure_ascii=False, indent=2)
        
        train_zero_anno_path = os.path.join(report_dir, "train_zero_anno.txt")
        with open(train_zero_anno_path, 'w', encoding='utf-8') as f:
            for img in train_zero_anno_images:
                f.write(f"{img}\n")
        
        val_zero_anno_path = os.path.join(report_dir, "val_zero_anno.txt")
        with open(val_zero_anno_path, 'w', encoding='utf-8') as f:
            for img in val_zero_anno_images:
                f.write(f"{img}\n")
    
    if has_issues:
        print("数据集存在以下问题:")
        
        if train_missing_images:
            print(f"- 训练集中有 {len(train_missing_images)} 张图片在标注中存在但在磁盘上不存在")
        
        if val_missing_images:
            print(f"- 验证集中有 {len(val_missing_images)} 张图片在标注中存在但在磁盘上不存在")
        
        if train_ext_mismatch:
            print(f"- 训练集中有 {len(train_ext_mismatch)} 张图片扩展名不匹配")
        
        if val_ext_mismatch:
            print(f"- 验证集中有 {len(val_ext_mismatch)} 张图片扩展名不匹配")
        
        if train_unlabeled_images:
            print(f"- 训练集中有 {len(train_unlabeled_images)} 张图片在磁盘上存在但未在标注中引用")
        
        if val_unlabeled_images:
            print(f"- 验证集中有 {len(val_unlabeled_images)} 张图片在磁盘上存在但未在标注中引用")
        
        if train_zero_anno_images:
            print(f"- 训练集中有 {len(train_zero_anno_images)} 张图片在标注中但没有任何实际标注")
        
        if val_zero_anno_images:
            print(f"- 验证集中有 {len(val_zero_anno_images)} 张图片在标注中但没有任何实际标注")
        
        if any(len(issues) > 0 for issues in train_bbox_issues.values()):
            print(f"- 训练集中存在边界框问题")
        
        if any(len(issues) > 0 for issues in val_bbox_issues.values()):
            print(f"- 验证集中存在边界框问题")
            
        # 修复建议
        print("\n可能的解决方案:")
        print("1. 确保标注文件中引用的所有图片在对应目录中存在")
        print("2. 修复文件扩展名不匹配的问题 (使用 --fix-ext 参数)")
        print("3. 移除没有标注的图片 (使用 --remove-empty 参数)")
        print("4. 检查并修复任何边界框问题")
        print("5. 确保file_name字段格式正确（是否需要包含路径前缀）")
        
        if save_report:
            print(f"\n所有问题文件列表已保存到 {report_dir} 目录")
    else:
        print("恭喜！数据集没有发现明显问题。")
    
    return has_issues, report_dir if save_report else None

def fix_extension_mismatches(data_root, train_anno_file, val_anno_file, report_dir):
    """修复标注文件中的扩展名不匹配问题"""
    train_anno_path = os.path.join(data_root, train_anno_file)
    val_anno_path = os.path.join(data_root, val_anno_file)
    
    # 加载扩展名不匹配信息
    train_ext_mismatch_path = os.path.join(report_dir, "train_ext_mismatch.json")
    val_ext_mismatch_path = os.path.join(report_dir, "val_ext_mismatch.json")
    
    if not os.path.exists(train_ext_mismatch_path) or not os.path.exists(val_ext_mismatch_path):
        print("错误: 未找到扩展名不匹配报告文件")
        return False
    
    with open(train_ext_mismatch_path, 'r', encoding='utf-8') as f:
        train_ext_mismatch = json.load(f)
    
    with open(val_ext_mismatch_path, 'r', encoding='utf-8') as f:
        val_ext_mismatch = json.load(f)
    
    # 加载标注文件
    with open(train_anno_path, 'r', encoding='utf-8') as f:
        train_anno = json.load(f)
    
    with open(val_anno_path, 'r', encoding='utf-8') as f:
        val_anno = json.load(f)
    
    # 创建备份
    train_backup_path = f"{train_anno_path}.bak"
    val_backup_path = f"{val_anno_path}.bak"
    
    if not os.path.exists(train_backup_path):
        with open(train_backup_path, 'w', encoding='utf-8') as f:
            json.dump(train_anno, f, ensure_ascii=False, indent=2)
    
    if not os.path.exists(val_backup_path):
        with open(val_backup_path, 'w', encoding='utf-8') as f:
            json.dump(val_anno, f, ensure_ascii=False, indent=2)
    
    # 修复训练集标注
    train_fixed_count = 0
    for i, img in enumerate(train_anno['images']):
        old_filename = os.path.basename(img['file_name'])
        if old_filename in train_ext_mismatch:
            new_filename = train_ext_mismatch[old_filename]
            img['file_name'] = new_filename
            train_fixed_count += 1
    
    # 修复验证集标注
    val_fixed_count = 0
    for i, img in enumerate(val_anno['images']):
        old_filename = os.path.basename(img['file_name'])
        if old_filename in val_ext_mismatch:
            new_filename = val_ext_mismatch[old_filename]
            img['file_name'] = new_filename
            val_fixed_count += 1
    
    # 保存修复后的标注文件
    with open(train_anno_path, 'w', encoding='utf-8') as f:
        json.dump(train_anno, f, ensure_ascii=False, indent=2)
    
    with open(val_anno_path, 'w', encoding='utf-8') as f:
        json.dump(val_anno, f, ensure_ascii=False, indent=2)
    
    print(f"扩展名修复完成：")
    print(f"- 训练集: 修复了 {train_fixed_count} 个文件名")
    print(f"- 验证集: 修复了 {val_fixed_count} 个文件名")
    print(f"- 原始标注文件已备份到 {train_backup_path} 和 {val_backup_path}")
    
    return True

def remove_empty_annotations(data_root, train_anno_file, val_anno_file, report_dir):
    """移除没有实际标注的图片条目"""
    train_anno_path = os.path.join(data_root, train_anno_file)
    val_anno_path = os.path.join(data_root, val_anno_file)
    
    # 加载没有标注的图片列表
    train_zero_anno_path = os.path.join(report_dir, "train_zero_anno.txt")
    val_zero_anno_path = os.path.join(report_dir, "val_zero_anno.txt")
    
    train_zero_anno = []
    if os.path.exists(train_zero_anno_path):
        with open(train_zero_anno_path, 'r', encoding='utf-8') as f:
            train_zero_anno = [line.strip() for line in f]
    
    val_zero_anno = []
    if os.path.exists(val_zero_anno_path):
        with open(val_zero_anno_path, 'r', encoding='utf-8') as f:
            val_zero_anno = [line.strip() for line in f]
    
    # 加载标注文件
    with open(train_anno_path, 'r', encoding='utf-8') as f:
        train_anno = json.load(f)
    
    with open(val_anno_path, 'r', encoding='utf-8') as f:
        val_anno = json.load(f)
    
    # 创建备份
    train_backup_path = f"{train_anno_path}.no_empty.bak"
    val_backup_path = f"{val_anno_path}.no_empty.bak"
    
    if not os.path.exists(train_backup_path):
        with open(train_backup_path, 'w', encoding='utf-8') as f:
            json.dump(train_anno, f, ensure_ascii=False, indent=2)
    
    if not os.path.exists(val_backup_path):
        with open(val_backup_path, 'w', encoding='utf-8') as f:
            json.dump(val_anno, f, ensure_ascii=False, indent=2)
    
    # 移除训练集中的空标注图片
    train_zero_anno_set = set(train_zero_anno)
    train_images_new = []
    for img in train_anno['images']:
        file_name = os.path.basename(img['file_name'])
        if file_name not in train_zero_anno_set:
            train_images_new.append(img)
    
    train_removed = len(train_anno['images']) - len(train_images_new)
    train_anno['images'] = train_images_new
    
    # 移除验证集中的空标注图片
    val_zero_anno_set = set(val_zero_anno)
    val_images_new = []
    for img in val_anno['images']:
        file_name = os.path.basename(img['file_name'])
        if file_name not in val_zero_anno_set:
            val_images_new.append(img)
    
    val_removed = len(val_anno['images']) - len(val_images_new)
    val_anno['images'] = val_images_new
    
    # 保存修改后的标注文件
    with open(train_anno_path, 'w', encoding='utf-8') as f:
        json.dump(train_anno, f, ensure_ascii=False, indent=2)
    
    with open(val_anno_path, 'w', encoding='utf-8') as f:
        json.dump(val_anno, f, ensure_ascii=False, indent=2)
    
    print(f"移除空标注完成：")
    print(f"- 训练集: 移除了 {train_removed} 个没有标注的图片")
    print(f"- 验证集: 移除了 {val_removed} 个没有标注的图片")
    print(f"- 原始标注文件已备份到 {train_backup_path} 和 {val_backup_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='验证COCO格式数据集的一致性并保存问题报告')
    parser.add_argument('--data_root', type=str, default='data/Bamberg_coco1024',
                       help='数据集根目录')
    parser.add_argument('--train_anno', type=str, default='annotations/instances_train2024.json',
                       help='训练集标注文件路径（相对于data_root）')
    parser.add_argument('--val_anno', type=str, default='annotations/instances_val2024.json',
                       help='验证集标注文件路径（相对于data_root）')
    parser.add_argument('--train_img_dir', type=str, default='train2024',
                       help='训练集图片目录（相对于data_root）')
    parser.add_argument('--val_img_dir', type=str, default='val2024',
                       help='验证集图片目录（相对于data_root）')
    parser.add_argument('--report_dir', type=str, default=None,
                       help='保存报告的目录（默认为data_root/validation_report）')
    parser.add_argument('--fix-ext', action='store_true',
                       help='修复标注文件中的扩展名不匹配问题')
    parser.add_argument('--remove-empty', action='store_true',
                       help='移除没有标注的图片条目')
    
    args = parser.parse_args()
    
    report_dir = args.report_dir if args.report_dir else os.path.join(args.data_root, 'validation_report')
    
    has_issues, report_path = validate_coco_dataset(
        data_root=args.data_root,
        train_anno_file=args.train_anno,
        val_anno_file=args.val_anno,
        train_img_dir=args.train_img_dir,
        val_img_dir=args.val_img_dir,
        save_report=True,
        report_dir=report_dir
    )
    
    if has_issues:
        if args.fix_ext:
            fix_extension_mismatches(args.data_root, args.train_anno, args.val_anno, report_path)
        
        if args.remove_empty:
            remove_empty_annotations(args.data_root, args.train_anno, args.val_anno, report_path)
    
    print("\n使用建议:")
    print("1. 先运行验证脚本: python check_coco_dataset.py")
    print("2. 修复扩展名问题: python check_coco_dataset.py --fix-ext")
    print("3. 移除空标注图片: python check_coco_dataset.py --remove-empty")
    print("4. 再次验证数据集: python check_coco_dataset.py")

if __name__ == "__main__":
    main()