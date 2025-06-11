import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
import argparse

def rename_forest_dataset(dataset_root, output_dir):
    """
    对森林损伤数据集中的垂直和倾斜图片及XML进行重命名，保持原始目录结构
    
    Args:
        dataset_root: 数据集根目录
        output_dir: 输出目录
    """
    # 设置处理的类别
    categories = ["vertical", "oblique"]
    
    # 创建输出基础目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 记录文件名映射
    mapping_data = []
    
    # 处理计数器
    total_renamed = 0
    total_errors = 0
    
    # 地点编码映射
    site_codes = {
        "Backsjon": "B",
        "Lidhem": "L",
        "Viken": "V"
    }
    
    # 遍历类别目录
    for category in categories:
        # 角度编码: V=垂直, O=倾斜
        angle_code = "V" if category == "vertical" else "O"
        
        # 创建类别输出目录
        category_output = os.path.join(output_dir, category)
        os.makedirs(category_output, exist_ok=True)
        
        category_path = os.path.join(dataset_root, category)
        if not os.path.exists(category_path):
            print(f"目录不存在: {category_path}")
            continue
        
        print(f"\n处理 {category} 类别...")
        
        # 获取所有调查文件夹
        site_folders = [f for f in os.listdir(category_path) 
                        if os.path.isdir(os.path.join(category_path, f))]
        
        # 遍历地点文件夹
        for site_folder in site_folders:
            # 解析地点名称
            site_name = site_folder.split('_')[0]
            site_code = site_codes.get(site_name, site_name[0].upper())
            
            # 保留原始文件夹名称
            site_output = os.path.join(category_output, site_folder)
            site_images_output = os.path.join(site_output, "Images")
            site_annotations_output = os.path.join(site_output, "Annotations")
            
            # 创建输出目录
            os.makedirs(site_images_output, exist_ok=True)
            os.makedirs(site_annotations_output, exist_ok=True)
            
            # 图片和标注路径
            images_folder = os.path.join(category_path, site_folder, "Images")
            annotations_folder = os.path.join(category_path, site_folder, "Annotations")
            
            if not os.path.exists(images_folder):
                print(f"图像目录不存在: {images_folder}")
                continue
                
            if not os.path.exists(annotations_folder):
                print(f"标注目录不存在: {annotations_folder}")
                continue
            
            # 获取图片列表
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png']:
                image_files.extend([f for f in os.listdir(images_folder) 
                                   if f.lower().endswith(ext)])
            
            print(f"找到 {site_folder} 的 {len(image_files)} 张图片...")
            
            # 初始化计数器
            counter = 1
            
            # 处理每张图片
            for img_file in tqdm(image_files, desc=f"{site_folder}"):
                base_name, ext = os.path.splitext(img_file)
                
                # 提取文件ID (使用UUID的前8位)
                file_id = base_name[:8] if len(base_name) >= 8 else base_name
                
                # 新文件名格式: 地点_角度_ID_序号.扩展名
                new_image_name = f"{site_code}_{angle_code}_{file_id}_{counter:04d}{ext}"
                new_xml_name = f"{site_code}_{angle_code}_{file_id}_{counter:04d}.xml"
                
                # 源文件路径
                src_img_path = os.path.join(images_folder, img_file)
                src_xml_path = os.path.join(annotations_folder, f"{base_name}.xml")
                
                # 目标文件路径
                dst_img_path = os.path.join(site_images_output, new_image_name)
                dst_xml_path = os.path.join(site_annotations_output, new_xml_name)
                
                try:
                    # 复制图片
                    shutil.copy2(src_img_path, dst_img_path)
                    
                    # 处理XML文件(如果存在)
                    if os.path.exists(src_xml_path):
                        # 读取XML
                        tree = ET.parse(src_xml_path)
                        root = tree.getroot()
                        
                        # 更新文件名
                        filename_elem = root.find('filename')
                        if filename_elem is not None:
                            filename_elem.text = new_image_name
                        
                        # 更新路径
                        path_elem = root.find('path')
                        if path_elem is not None:
                            path_elem.text = os.path.join("Images", new_image_name)
                        
                        # 保存更新后的XML
                        tree.write(dst_xml_path, encoding='utf-8', xml_declaration=True)
                        
                        # 记录映射
                        mapping_data.append({
                            "原目录": site_folder,
                            "原图片名": img_file,
                            "原XML名": f"{base_name}.xml",
                            "新图片名": new_image_name,
                            "新XML名": new_xml_name,
                            "地点": site_name,
                            "角度": category
                        })
                        
                        total_renamed += 1
                    else:
                        print(f"警告: 未找到对应的XML文件: {src_xml_path}")
                        total_errors += 1
                
                except Exception as e:
                    print(f"处理文件时出错 ({img_file}): {str(e)}")
                    total_errors += 1
                
                counter += 1
    
    # 保存映射文件
    if mapping_data:
        mapping_df = pd.DataFrame(mapping_data)
        mapping_path = os.path.join(output_dir, "file_mapping.csv")
        mapping_df.to_csv(mapping_path, index=False, encoding='utf-8')
        
        # 输出按地点和角度的统计
        stats_df = mapping_df.groupby(['地点', '角度']).size().reset_index(name='文件数')
        print("\n重命名统计:")
        print(stats_df.to_string(index=False))
    
    # 输出结果
    print(f"\n处理完成!")
    print(f"- 总共重命名: {total_renamed} 对文件")
    print(f"- 处理错误: {total_errors}")
    print(f"- 重命名后的数据保存在: {output_dir}")
    print(f"- 文件映射保存在: {os.path.join(output_dir, 'file_mapping.csv')}")
    print(f"\n保持了原始目录结构:")
    for category in categories:
        print(f"- {os.path.join(output_dir, category)}/")
        site_folders = [f for f in os.listdir(os.path.join(dataset_root, category)) 
                         if os.path.isdir(os.path.join(dataset_root, category, f))]
        for site in site_folders:
            print(f"  └── {site}/")
            print(f"      ├── Images/")
            print(f"      └── Annotations/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重命名森林损伤数据集")
    parser.add_argument("--dataset", default="./Data_Set_Spruce_Bark_Beetle",
                        help="数据集根目录")
    parser.add_argument("--output", default="./Renamed_Forest_Dataset",
                        help="输出目录")
    
    args = parser.parse_args()
    rename_forest_dataset(args.dataset, args.output)