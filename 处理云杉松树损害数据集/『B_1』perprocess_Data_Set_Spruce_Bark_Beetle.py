import os
import xml.etree.ElementTree as ET
from PIL import Image
import glob
from tqdm import tqdm

def fix_xml_annotation(xml_path, img_path, min_area=50, backup=False):
    """
    修复XML标注文件中超出图片范围的边界框，并删除面积小的和无效的框
    
    Args:
        xml_path: XML文件路径
        img_path: 对应的图片路径
        min_area: 最小边界框面积阈值
        backup: 是否备份原文件
    
    Returns:
        fixed: 是否进行了修复
    """
    try:
        # 获取图片尺寸
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # 解析XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        fixed = False
        objects_to_remove = []
        
        # 更新size标签
        size_elem = root.find('size')
        if size_elem is not None:
            width_elem = size_elem.find('width')
            height_elem = size_elem.find('height')
            
            if width_elem is not None and height_elem is not None:
                width_elem.text = str(img_width)
                height_elem.text = str(img_height)
                fixed = True
        
        # 检查并修复每个目标的边界框
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is None:
                # 边界框不存在，标记为删除
                objects_to_remove.append(obj)
                fixed = True
                continue
                
            try:
                # 获取边界框坐标
                xmin_elem = bndbox.find('xmin')
                ymin_elem = bndbox.find('ymin')
                xmax_elem = bndbox.find('xmax')
                ymax_elem = bndbox.find('ymax')
                
                if None in (xmin_elem, ymin_elem, xmax_elem, ymax_elem):
                    # 边界框坐标不完整，标记为删除
                    objects_to_remove.append(obj)
                    fixed = True
                    continue
                    
                xmin = float(xmin_elem.text)
                ymin = float(ymin_elem.text)
                xmax = float(xmax_elem.text)
                ymax = float(ymax_elem.text)
                
                # 检查坐标有效性
                if xmin >= xmax or ymin >= ymax:
                    # 无效边界框，标记为删除
                    objects_to_remove.append(obj)
                    fixed = True
                    continue
                
                # 修复超出范围的坐标
                original_xmin, original_ymin = xmin, ymin
                original_xmax, original_ymax = xmax, ymax
                
                if xmin < 0:
                    xmin = 0
                    fixed = True
                
                if ymin < 0:
                    ymin = 0
                    fixed = True
                
                if xmax > img_width:
                    xmax = img_width
                    fixed = True
                
                if ymax > img_height:
                    ymax = img_height
                    fixed = True
                
                # 计算面积并检查
                area = (xmax - xmin) * (ymax - ymin)
                if area < min_area:
                    # 面积过小，标记为删除
                    objects_to_remove.append(obj)
                    fixed = True
                    continue
                
                # 如果坐标有修改，更新XML
                if (xmin != original_xmin or ymin != original_ymin or 
                    xmax != original_xmax or ymax != original_ymax):
                    xmin_elem.text = str(int(xmin))
                    ymin_elem.text = str(int(ymin))
                    xmax_elem.text = str(int(xmax))
                    ymax_elem.text = str(int(ymax))
                    fixed = True
                    
            except (ValueError, AttributeError):
                # 解析错误，标记为删除
                objects_to_remove.append(obj)
                fixed = True
                continue
        
        # 删除标记的对象
        for obj in objects_to_remove:
            root.remove(obj)
        
        # 如果有修改，保存文件
        if fixed:
            if backup:
                backup_path = xml_path + '.backup'
                if not os.path.exists(backup_path):
                    tree.write(backup_path, encoding='utf-8')
            
            tree.write(xml_path, encoding='utf-8')
            
        return fixed
        
    except Exception as e:
        print(f"处理文件出错 {xml_path}: {str(e)}")
        return False

def process_directory(directory, min_area=10):
    """处理单个目录下的所有XML文件"""
    print(f"\n处理目录: {os.path.basename(directory)}")
    
    images_dir = os.path.join(directory, "Images")
    annotations_dir = os.path.join(directory, "Annotations")
    
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        print(f"错误: 目录结构不完整 - {directory}")
        return 0, 0
        
    # 获取所有图片
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
        
    total_files = len(image_files)
    fixed_files = 0
    
    for image_file in tqdm(image_files, desc="处理文件"):
        basename = os.path.splitext(os.path.basename(image_file))[0]
        xml_file = os.path.join(annotations_dir, f"{basename}.xml")
        
        if os.path.exists(xml_file):
            if fix_xml_annotation(xml_file, image_file, min_area):
                fixed_files += 1
                
    print(f"处理完成: 修复了 {fixed_files}/{total_files} 个文件")
    
    return fixed_files, total_files

def process_all_data(dataset_path, min_area=10):
    """处理数据集中的所有数据"""
    print("=== 开始修复标注文件 ===")
    print(f"最小边界框面积阈值: {min_area} 像素")
    
    # 获取垂直和倾斜拍摄的子目录
    vertical_dirs = glob.glob(os.path.join(dataset_path, "vertical", "*"))
    oblique_dirs = glob.glob(os.path.join(dataset_path, "oblique", "*"))
    
    vertical_dirs = [d for d in vertical_dirs if os.path.isdir(d)]
    oblique_dirs = [d for d in oblique_dirs if os.path.isdir(d)]
    
    print(f"\n找到 {len(vertical_dirs)} 个垂直拍摄目录和 {len(oblique_dirs)} 个倾斜拍摄目录")
    
    total_fixed = 0
    total_files = 0
    
    # 处理垂直拍摄数据
    print("\n处理垂直拍摄数据...")
    for dir_path in vertical_dirs:
        fixed, total = process_directory(dir_path, min_area)
        total_fixed += fixed
        total_files += total
        
    # 处理倾斜拍摄数据
    print("\n处理倾斜拍摄数据...")
    for dir_path in oblique_dirs:
        fixed, total = process_directory(dir_path, min_area)
        total_fixed += fixed
        total_files += total
        
    print("\n=== 处理完成 ===")
    print(f"总文件数: {total_files}")
    print(f"修复文件数: {total_fixed}")

if __name__ == "__main__":
    # 指定数据集路径
    dataset_path = "../data/Data_Set_Spruce_Bark_Beetle"
    # 设置最小边界框面积阈值
    min_area = 50
    process_all_data(dataset_path, min_area)