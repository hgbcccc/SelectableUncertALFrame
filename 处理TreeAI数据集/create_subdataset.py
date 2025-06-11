import os
import shutil
from collections import defaultdict

# 配置参数（注意：这里的类别ID是文档中的原始ID）
target_class_ids = [1, 2, 3, 4, 5, 6, 7]  # 文档中的类别ID

# 原始数据集路径
original_train_img_dir = '../data/12_RGB_FullyLabeled_640/coco/train/images'
original_train_label_dir = '../data/12_RGB_FullyLabeled_640/coco/train/labels'
original_val_img_dir = '../data/12_RGB_FullyLabeled_640/coco/val/images'
original_val_label_dir = '../data/12_RGB_FullyLabeled_640/coco/val/labels'

# 新数据集路径
new_dataset_root = 'TreeAI'
os.makedirs(new_dataset_root, exist_ok=True)

def filter_dataset(original_img_dir, original_label_dir, new_img_dir, new_label_dir):
    """
    筛选指定类别的数据并复制到新目录
    """
    os.makedirs(new_img_dir, exist_ok=True)
    os.makedirs(new_label_dir, exist_ok=True)
    
    selected_files = set()
    class_stats = defaultdict(int)
    
    # 第一步：筛选包含目标类别的标签文件
    for label_file in os.listdir(original_label_dir):
        label_path = os.path.join(original_label_dir, label_file)
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        # 注意：标签文件中类别ID从0开始，文档中从1开始
                        class_id_in_file = int(line.split()[0])
                        # 转换为文档中的ID（+1）
                        doc_class_id = class_id_in_file + 1
                        if doc_class_id in target_class_ids:
                            selected_files.add(label_file)
                            class_stats[doc_class_id] += 1
                            break  # 只要包含一个目标类别就保留
                    except (ValueError, IndexError):
                        continue
    
    # 第二步：复制对应的图像和标签文件
    for base_name in selected_files:
        # 处理标签文件
        src_label = os.path.join(original_label_dir, base_name)
        dst_label = os.path.join(new_label_dir, base_name)
        
        # 处理图像文件（尝试多种图像扩展名）
        img_base = os.path.splitext(base_name)[0]
        for ext in ['.jpg', '.jpeg', '.png']:  # 尝试常见图像格式
            src_img = os.path.join(original_img_dir, img_base + ext)
            if os.path.exists(src_img):
                dst_img = os.path.join(new_img_dir, img_base + ext)
                shutil.copy2(src_img, dst_img)
                break
        
        # 过滤标签文件，只保留目标类别
        with open(src_label, 'r') as src_f, open(dst_label, 'w') as dst_f:
            for line in src_f:
                line = line.strip()
                if line:
                    try:
                        class_id_in_file = int(line.split()[0])
                        doc_class_id = class_id_in_file + 1
                        if doc_class_id in target_class_ids:
                            dst_f.write(line + '\n')
                    except (ValueError, IndexError):
                        continue
    
    return class_stats

print("正在处理训练集...")
train_stats = filter_dataset(original_train_img_dir, original_train_label_dir, 
                           os.path.join(new_dataset_root, 'train/images'),
                           os.path.join(new_dataset_root, 'train/labels'))

print("\n正在处理验证集...")
val_stats = filter_dataset(original_val_img_dir, original_val_label_dir,
                         os.path.join(new_dataset_root, 'val/images'),
                         os.path.join(new_dataset_root, 'val/labels'))

# 打印统计信息
print("\n===== 数据集统计 =====")
print("训练集类别分布:")
for class_id, count in sorted(train_stats.items()):
    print(f"  文档ID {class_id}: {count}个样本")

print("\n验证集类别分布:")
for class_id, count in sorted(val_stats.items()):
    print(f"  文档ID {class_id}: {count}个样本")

print(f"\n新数据集已创建到: {new_dataset_root}")