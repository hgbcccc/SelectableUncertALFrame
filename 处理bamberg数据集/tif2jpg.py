from PIL import Image, ImageFile
import os
import json
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def convert_tif_to_jpg(source_folder, target_folder, error_log_file=None):
    """
    将TIF文件转换为JPG格式并记录错误到文本文件
    
    参数:
    source_folder: 源文件夹路径
    target_folder: 目标文件夹路径
    error_log_file: 错误日志文件路径，如果为None则不保存错误日志
    
    返回:
    success_count: 成功转换的文件数
    error_count: 失败的文件数
    error_files: 失败文件列表
    """
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 获取所有需要转换的文件列表
    tif_files = [f for f in os.listdir(source_folder) if f.endswith('.tif') or f.endswith('.tiff')]
    
    # 统计信息
    success_count = 0
    error_count = 0
    error_files = []
    
    # 允许加载截断的图像
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    # 使用tqdm创建进度条
    for filename in tqdm(tif_files, desc="转换TIF到JPG", unit="文件"):
        # 构建完整的文件路径
        file_path = os.path.join(source_folder, filename)
        # 构建目标文件路径
        target_path = os.path.join(target_folder, filename[:-4] + '.jpg')
        
        try:
            # 打开图像文件
            with Image.open(file_path) as img:
                # 转换图像格式并保存
                img.convert('RGB').save(target_path, 'JPEG', quality=90)
                success_count += 1
        except Exception as e:
            error_count += 1
            error_files.append((filename, str(e)))
            print(f"\n警告: 处理文件 {filename} 时出错: {e}")
    
    print(f"完成转换！共处理 {len(tif_files)} 个文件")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {error_count} 个文件")
    
    # 将错误信息保存到文本文件
    if error_log_file and error_count > 0:
        with open(error_log_file, 'w', encoding='utf-8') as f:
            f.write(f"转换失败文件列表 ({error_count}/{len(tif_files)}):\n")
            f.write(f"源文件夹: {source_folder}\n")
            f.write(f"目标文件夹: {target_folder}\n")
            f.write("-" * 80 + "\n")
            for i, (file, error) in enumerate(error_files, 1):
                f.write(f"{i}. {file}: {error}\n")
        
        print(f"错误信息已保存到: {error_log_file}")
    
    return success_count, error_count, error_files

def update_json_file_references(json_file_path, output_json_path=None, error_files=None):
    """
    更新JSON文件中的图像文件引用，将.tif/.tiff替换为.jpg
    
    参数:
    json_file_path: 原始JSON文件路径
    output_json_path: 输出JSON文件路径，如果为None，则覆盖原文件
    error_files: 转换失败的文件列表，这些文件将被从JSON中移除
    
    返回:
    updated_count: 更新的文件数量
    removed_count: 移除的文件数量
    """
    if output_json_path is None:
        output_json_path = json_file_path
    
    # 转换错误文件列表为文件名集合
    error_filenames = set()
    if error_files:
        error_filenames = {file[0] for file in error_files}
    
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 修改图像文件名引用并记录需要移除的图像ID
    updated_count = 0
    removed_count = 0
    removed_image_ids = set()
    
    if 'images' in data:
        # 创建新的图像列表
        new_images = []
        
        for image in tqdm(data['images'], desc="更新JSON文件引用", unit="项"):
            if 'file_name' in image:
                filename = os.path.basename(image['file_name'])
                
                # 检查是否是失败转换的文件
                if filename in error_filenames:
                    removed_image_ids.add(image['id'])
                    removed_count += 1
                    continue
                
                # 更新扩展名
                if filename.endswith('.tif') or filename.endswith('.tiff'):
                    # 获取文件名（不含扩展名）
                    file_name_base = os.path.splitext(image['file_name'])[0]
                    # 更新为.jpg扩展名
                    image['file_name'] = file_name_base + '.jpg'
                    updated_count += 1
                
                # 保留这个图像
                new_images.append(image)
        
        # 更新图像列表
        data['images'] = new_images
    
    # 如果有需要移除的图像，也移除相应的标注
    removed_annotations = 0
    if removed_image_ids and 'annotations' in data:
        original_count = len(data['annotations'])
        new_annotations = [ann for ann in data['annotations'] 
                          if ann.get('image_id') not in removed_image_ids]
        
        removed_annotations = original_count - len(new_annotations)
        data['annotations'] = new_annotations
        print(f"移除了 {removed_annotations} 个与失败图像相关的标注")
    
    # 保存更新后的JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"JSON文件更新完成！")
    print(f"- 更新了 {updated_count} 个文件引用")
    print(f"- 移除了 {removed_count} 个无效图像")
    print(f"- 移除了 {removed_annotations} 个相关标注")
    print(f"- 保存至 {output_json_path}")
    
    return updated_count, removed_count

def validate_coco_dataset(ann_file, img_dir, num_samples=5, output_dir=None):
    """
    使用COCO API验证数据集，并可视化部分样本的标注
    
    参数:
    ann_file: COCO标注文件路径
    img_dir: 图像目录路径
    num_samples: 要可视化的样本数量
    output_dir: 可视化结果输出目录，如果为None则显示而不保存
    
    返回:
    None
    """
    print(f"加载COCO数据集: {ann_file}")
    
    # 使用COCO API加载数据集
    try:
        coco = COCO(ann_file)
    except Exception as e:
        print(f"加载COCO数据集失败: {e}")
        return
    
    # 获取数据集统计信息
    img_ids = coco.getImgIds()
    num_images = len(img_ids)
    ann_ids = coco.getAnnIds()
    num_annotations = len(ann_ids)
    categories = coco.loadCats(coco.getCatIds())
    
    print("\nCOCO数据集统计:")
    print(f"图像数量: {num_images}")
    print(f"标注数量: {num_annotations}")
    print(f"每张图像平均标注数: {num_annotations/max(1, num_images):.2f}")
    print(f"类别数量: {len(categories)}")
    print("\n类别列表:")
    for cat in categories:
        ann_count = len(coco.getAnnIds(catIds=[cat['id']]))
        print(f"  - {cat['name']}: {ann_count} 个标注 (ID: {cat['id']})")
    
    # 确保输出目录存在
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 随机选择样本进行可视化
    if num_images > 0:
        try:
            # 随机选择图像ID
            sample_img_ids = np.random.choice(img_ids, min(num_samples, num_images), replace=False)
            
            for i, img_id in enumerate(sample_img_ids):
                # 加载图像信息
                img_infos = coco.loadImgs(img_id)
                
                # 检查是否有有效的图像信息
                if not img_infos or len(img_infos) == 0:
                    print(f"警告: 无法获取图像ID为 {img_id} 的信息，跳过")
                    continue
                    
                img_info = img_infos[0]
                img_path = os.path.join(img_dir, img_info['file_name'])
                
                # 检查图像是否存在
                if not os.path.exists(img_path):
                    print(f"警告: 图像文件不存在: {img_path}")
                    continue
                    
                # 读取图像
                try:
                    img = plt.imread(img_path)
                except Exception as e:
                    print(f"警告: 无法读取图像 {img_path}: {e}")
                    continue
                
                # 获取该图像的所有标注
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                
                # 创建画布
                fig, ax = plt.subplots(1, figsize=(12, 8))
                ax.imshow(img)
                ax.set_title(f"图像 {img_info['file_name']} (ID: {img_id}), {len(anns)} 个标注")
                
                # 绘制每个标注的边界框
                colors = plt.cm.hsv(np.linspace(0, 1, len(categories) + 1))
                
                for ann in anns:
                    # 获取分类信息
                    cat_id = ann['category_id']
                    cat_infos = coco.loadCats([cat_id])
                    if not cat_infos:
                        print(f"警告: 无法获取类别ID为 {cat_id} 的信息，跳过标注")
                        continue
                    
                    cat_info = cat_infos[0]
                    cat_idx = next((i for i, c in enumerate(categories) if c['id'] == cat_id), 0)
                    
                    # 绘制边界框
                    if 'bbox' in ann:
                        bbox = ann['bbox']  # [x, y, width, height]
                        rect = patches.Rectangle(
                            (bbox[0], bbox[1]), bbox[2], bbox[3],
                            linewidth=2, edgecolor=colors[cat_idx % len(colors)],
                            facecolor='none', alpha=0.7
                        )
                        ax.add_patch(rect)
                        
                        # 添加标签
                        label = f"{cat_info['name']} (ID: {cat_id})"
                        ax.text(
                            bbox[0], bbox[1] - 5, label,
                            color='white', fontsize=8, 
                            bbox={'facecolor': colors[cat_idx % len(colors)], 'alpha': 0.7, 'pad': 1}
                        )
                
                # 调整显示
                plt.tight_layout()
                
                # 保存或显示图像
                if output_dir:
                    output_path = os.path.join(output_dir, f"sample_{i+1}_{os.path.basename(img_info['file_name'])}")
                    plt.savefig(output_path, dpi=100, bbox_inches='tight')
                    print(f"保存验证图像: {output_path}")
                    plt.close(fig)
                else:
                    plt.show()
        except Exception as e:
            print(f"可视化过程中出现错误: {e}")
    
    print("\nCOCO数据集验证完成")

# 指定源文件夹和目标文件夹
source_folder = 'data/Bamberg_coco1024/val2024'
target_folder = 'data/Bamberg_coco1024/val2024_jpg'

source_folder1 = 'data/Bamberg_coco1024/train2024'
target_folder1 = 'data/Bamberg_coco1024/train2024_jpg'

# 指定JSON文件路径
val_json_path = 'data/Bamberg_coco1024/annotations/instances_val2024.json'
train_json_path = 'data/Bamberg_coco1024/annotations/instances_train2024.json'

# 更新后的JSON文件路径
val_json_path_updated = 'data/Bamberg_coco1024/annotations/instances_val2024_jpg.json'
train_json_path_updated = 'data/Bamberg_coco1024/annotations/instances_train2024_jpg.json'

# 错误日志文件路径
val_error_log = 'data/Bamberg_coco1024/val2024_conversion_errors.txt'
train_error_log = 'data/Bamberg_coco1024/train2024_conversion_errors.txt'

# 可视化输出目录
validation_output_dir = 'data/Bamberg_coco1024/validation_samples'

# 根据需要注释掉已经完成的步骤
# 转换图像格式
print("处理训练集图像...")
_, _, train_errors = convert_tif_to_jpg(source_folder1, target_folder1, train_error_log)

print("\n处理验证集图像...")
_, _, val_errors = convert_tif_to_jpg(source_folder, target_folder, val_error_log)

# 更新JSON文件
print("\n更新训练集JSON文件...")
update_json_file_references(train_json_path, train_json_path_updated, train_errors)

print("\n更新验证集JSON文件...")
update_json_file_references(val_json_path, val_json_path_updated, val_errors)

# 验证转换后的数据集
print("\n验证训练集数据...")
validate_coco_dataset(
    train_json_path_updated, 
    target_folder1, 
    num_samples=3, 
    output_dir=os.path.join(validation_output_dir, 'train')
)

print("\n验证验证集数据...")
validate_coco_dataset(
    val_json_path_updated, 
    target_folder, 
    num_samples=3, 
    output_dir=os.path.join(validation_output_dir, 'val')
)

print("\n全部处理完成！")