import os
import xml.etree.ElementTree as ET
from PIL import Image
import glob
from tqdm import tqdm
import csv

def check_xml_annotation(xml_path, img_path):
    """
    检查XML标注文件中的边界框是否有问题
    
    Args:
        xml_path: XML文件路径
        img_path: 对应的图片路径
    
    Returns:
        issues: 发现的问题列表，每个问题是一个元组 (问题类型, 问题描述)
    """
    issues = []
    
    try:
        # 获取图片尺寸
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # 解析XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 检查size标签
        size_elem = root.find('size')
        if size_elem is not None:
            xml_width = int(size_elem.find('width').text)
            xml_height = int(size_elem.find('height').text)
            
            if xml_width != img_width or xml_height != img_height:
                issues.append(('size_mismatch', 
                    f"尺寸不匹配 - XML: {xml_width}x{xml_height}, 图片: {img_width}x{img_height}"))
        
        # 检查每个目标的边界框
        for i, obj in enumerate(root.findall('object')):
            name_elem = obj.find('name')
            class_name = "Unknown"
            if name_elem is not None and name_elem.text is not None:
                class_name = name_elem.text
                
            bndbox = obj.find('bndbox')
            if bndbox is None:
                issues.append(('missing_bbox', f"目标 {i} ({class_name}): 缺少边界框"))
                continue
                
            try:
                # 获取边界框坐标
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # 检查坐标有效性
                if xmin >= xmax or ymin >= ymax:
                    issues.append(('invalid_bbox', 
                        f"目标 {i} ({class_name}): 无效边界框 ({xmin},{ymin},{xmax},{ymax})"))
                    continue
                
                # 检查是否超出范围
                if xmin < 0:
                    issues.append(('out_of_bounds', 
                        f"目标 {i} ({class_name}): xmin ({xmin}) < 0"))
                
                if ymin < 0:
                    issues.append(('out_of_bounds', 
                        f"目标 {i} ({class_name}): ymin ({ymin}) < 0"))
                
                if xmax > img_width:
                    issues.append(('out_of_bounds', 
                        f"目标 {i} ({class_name}): xmax ({xmax}) > 图片宽度 ({img_width})"))
                
                if ymax > img_height:
                    issues.append(('out_of_bounds', 
                        f"目标 {i} ({class_name}): ymax ({ymax}) > 图片高度 ({img_height})"))
                
                # 检查面积
                area = (xmax - xmin) * (ymax - ymin)
                if area < 50:
                    issues.append(('small_area', 
                        f"目标 {i} ({class_name}): 面积过小 ({area:.1f} 像素)"))
                    
            except (ValueError, AttributeError) as e:
                issues.append(('parse_error', 
                    f"目标 {i} ({class_name}): 解析错误 - {str(e)}"))
                
        return issues
        
    except Exception as e:
        return [('file_error', f"处理文件出错: {str(e)}")]

def validate_directory(directory, output_csv=None):
    """检查目录中的所有XML文件"""
    print(f"\n验证目录: {os.path.basename(directory)}")
    
    images_dir = os.path.join(directory, "Images")
    annotations_dir = os.path.join(directory, "Annotations")
    
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        print(f"错误: 目录结构不完整 - {directory}")
        return None
        
    # 获取所有图片
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
        
    total_files = len(image_files)
    files_with_issues = 0
    total_issues = 0
    
    # 按问题类型统计
    issue_stats = {
        'out_of_bounds': 0,
        'small_area': 0,
        'invalid_bbox': 0,
        'missing_bbox': 0,
        'size_mismatch': 0,
        'parse_error': 0,
        'file_error': 0
    }
    
    # 保存问题详情
    all_issues = []
    
    for image_file in tqdm(image_files, desc="检查文件"):
        basename = os.path.splitext(os.path.basename(image_file))[0]
        xml_file = os.path.join(annotations_dir, f"{basename}.xml")
        
        if os.path.exists(xml_file):
            issues = check_xml_annotation(xml_file, image_file)
            
            if issues:
                files_with_issues += 1
                total_issues += len(issues)
                
                for issue_type, issue_desc in issues:
                    issue_stats[issue_type] = issue_stats.get(issue_type, 0) + 1
                    
                    all_issues.append([
                        os.path.basename(directory),
                        os.path.basename(xml_file),
                        os.path.basename(image_file),
                        issue_type,
                        issue_desc
                    ])
    
    # 输出统计信息
    print(f"检查完成: {files_with_issues}/{total_files} 个文件有问题，共 {total_issues} 个问题")
    print("问题类型统计:")
    for issue_type, count in issue_stats.items():
        if count > 0:
            print(f"- {issue_type}: {count}")
    
    # 写入CSV
    if output_csv and all_issues:
        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 如果文件为空，写入标题行
            if os.stat(output_csv).st_size == 0:
                writer.writerow(["目录", "XML文件", "图片文件", "问题类型", "问题描述"])
            writer.writerows(all_issues)
    
    return {
        'total_files': total_files,
        'files_with_issues': files_with_issues,
        'total_issues': total_issues,
        'issue_stats': issue_stats
    }

def validate_all_data(dataset_path, output_dir="validation_results"):
    """检查数据集中的所有数据"""
    print("=== 开始验证标注文件 ===")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备CSV文件
    output_csv = os.path.join(output_dir, "annotation_issues.csv")
    # 创建空文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        pass
    
    # 获取垂直和倾斜拍摄的子目录
    vertical_dirs = glob.glob(os.path.join(dataset_path, "vertical", "*"))
    oblique_dirs = glob.glob(os.path.join(dataset_path, "oblique", "*"))
    
    vertical_dirs = [d for d in vertical_dirs if os.path.isdir(d)]
    oblique_dirs = [d for d in oblique_dirs if os.path.isdir(d)]
    
    print(f"\n找到 {len(vertical_dirs)} 个垂直拍摄目录和 {len(oblique_dirs)} 个倾斜拍摄目录")
    
    # 总体统计
    overall_stats = {
        'total_files': 0,
        'files_with_issues': 0,
        'total_issues': 0,
        'issue_stats': {
            'out_of_bounds': 0,
            'small_area': 0,
            'invalid_bbox': 0,
            'missing_bbox': 0,
            'size_mismatch': 0,
            'parse_error': 0,
            'file_error': 0
        }
    }
    
    # 处理垂直拍摄数据
    print("\n验证垂直拍摄数据...")
    for dir_path in vertical_dirs:
        stats = validate_directory(dir_path, output_csv)
        if stats:
            overall_stats['total_files'] += stats['total_files']
            overall_stats['files_with_issues'] += stats['files_with_issues']
            overall_stats['total_issues'] += stats['total_issues']
            for issue_type, count in stats['issue_stats'].items():
                overall_stats['issue_stats'][issue_type] += count
    
    # 处理倾斜拍摄数据
    print("\n验证倾斜拍摄数据...")
    for dir_path in oblique_dirs:
        stats = validate_directory(dir_path, output_csv)
        if stats:
            overall_stats['total_files'] += stats['total_files']
            overall_stats['files_with_issues'] += stats['files_with_issues']
            overall_stats['total_issues'] += stats['total_issues']
            for issue_type, count in stats['issue_stats'].items():
                overall_stats['issue_stats'][issue_type] += count
    
    # 输出总体统计
    print("\n=== 验证完成 ===")
    print(f"总文件数: {overall_stats['total_files']}")
    print(f"有问题的文件数: {overall_stats['files_with_issues']}")
    print(f"总问题数: {overall_stats['total_issues']}")
    print("\n问题类型统计:")
    for issue_type, count in overall_stats['issue_stats'].items():
        if count > 0:
            print(f"- {issue_type}: {count}")
    
    # 保存统计报告
    report_file = os.path.join(output_dir, "validation_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== 标注验证报告 ===\n\n")
        f.write(f"总文件数: {overall_stats['total_files']}\n")
        f.write(f"有问题的文件数: {overall_stats['files_with_issues']}\n")
        f.write(f"总问题数: {overall_stats['total_issues']}\n\n")
        f.write("问题类型统计:\n")
        for issue_type, count in overall_stats['issue_stats'].items():
            if count > 0:
                f.write(f"- {issue_type}: {count}\n")
    
    print(f"\n详细问题列表已保存到: {output_csv}")
    print(f"统计报告已保存到: {report_file}")

if __name__ == "__main__":
    # 指定数据集路径
    dataset_path = "../data/Data_Set_Spruce_Bark_Beetle"
    validate_all_data(dataset_path)