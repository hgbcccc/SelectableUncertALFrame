import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import skimage.io as io
from matplotlib.patches import Rectangle
import random
from collections import Counter
import matplotlib as mpl

def validate_coco_dataset(annotation_file):
    """
    使用pycocotools验证COCO数据集
    
    Args:
        annotation_file: 标注文件路径
    """
    print(f"=== Validating COCO dataset: {annotation_file} ===")
    
    # 加载COCO API
    try:
        coco = COCO(annotation_file)
        print("✓ COCO dataset successfully loaded")
    except Exception as e:
        print(f"✗ Failed to load COCO dataset: {str(e)}")
        return None
    
    # 检查数据集内容
    cats = coco.loadCats(coco.getCatIds())
    print(f"\nCategories: {len(cats)}")
    for cat in cats:
        print(f"- ID: {cat['id']}, Name: {cat['name']}")
    
    img_ids = coco.getImgIds()
    print(f"\nImages: {len(img_ids)}")
    
    ann_ids = coco.getAnnIds()
    print(f"Annotations: {len(ann_ids)}")
    
    if len(ann_ids) == 0:
        print("✗ Error: No annotations in dataset!")
        return None
    
    # 每张图片的标注数量
    img_to_anns = {}
    for img_id in img_ids:
        anns = coco.getAnnIds(imgIds=img_id)
        img_to_anns[img_id] = len(anns)
    
    # 每个类别的标注数量
    cat_to_anns = {}
    for cat in cats:
        anns = coco.getAnnIds(catIds=cat['id'])
        cat_to_anns[cat['name']] = len(anns)
    
    # 标注统计信息
    ann_counts = Counter(img_to_anns.values())
    empty_images = ann_counts.get(0, 0)
    print(f"\nAnnotation statistics:")
    print(f"- Images without annotations: {empty_images} ({empty_images/len(img_ids)*100:.1f}%)")
    print(f"- Max annotations per image: {max(img_to_anns.values()) if img_to_anns else 0}")
    print(f"- Average annotations per image: {np.mean(list(img_to_anns.values())) if img_to_anns else 0:.2f}")
    
    print(f"\nCategory statistics:")
    for cat_name, count in cat_to_anns.items():
        print(f"- {cat_name}: {count} ({count/len(ann_ids)*100:.1f}%)")
    
    # 边界框尺寸统计
    areas = []
    widths = []
    heights = []
    aspect_ratios = []
    
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        bbox = ann['bbox']  # [x, y, width, height]
        area = bbox[2] * bbox[3]
        areas.append(area)
        widths.append(bbox[2])
        heights.append(bbox[3])
        aspect_ratios.append(bbox[2]/bbox[3] if bbox[3] > 0 else 0)
    
    print(f"\nBounding box statistics:")
    print(f"- Min area: {min(areas):.1f}")
    print(f"- Max area: {max(areas):.1f}")
    print(f"- Mean area: {np.mean(areas):.1f}")
    print(f"- Median area: {np.median(areas):.1f}")
    
    # 检查特殊字段
    has_damage_info = False
    if anns and 'damage_type' in anns[0]:
        has_damage_info = True
        damage_types = Counter([ann.get('damage_type', 'unknown') for ann in anns])
        print(f"\nDamage type statistics:")
        for damage, count in damage_types.items():
            print(f"- {damage}: {count} ({count/len(anns)*100:.1f}%)")
    
    return {
        "coco": coco,
        "img_ids": img_ids,
        "cats": cats,
        "anns": anns,
        "areas": areas,
        "widths": widths,
        "heights": heights,
        "aspect_ratios": aspect_ratios,
        "has_damage_info": has_damage_info
    }

def visualize_coco_dataset(coco_info, output_dir, num_samples=5):
    """
    可视化COCO数据集的样本图片和标注
    
    Args:
        coco_info: 验证函数返回的信息
        output_dir: 输出目录
        num_samples: 可视化的样本数量
    """
    if coco_info is None:
        print("Cannot visualize, validation failed")
        return
    
    coco = coco_info["coco"]
    img_ids = coco_info["img_ids"]
    cats = coco_info["cats"]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置matplotlib使用Agg后端，避免字体问题
    mpl.use('Agg')
    
    # 设置西文字体
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 生成一些可视化
    plt.figure(figsize=(18, 6))
    
    # 面积分布直方图
    plt.subplot(1, 3, 1)
    plt.hist(coco_info["areas"], bins=50)
    plt.title('Bbox Area Distribution')
    plt.xlabel('Area')
    plt.ylabel('Count')
    plt.yscale('log')
    
    # 宽高比分布
    plt.subplot(1, 3, 2)
    plt.hist(coco_info["aspect_ratios"], bins=50, range=(0, 5))
    plt.title('Bbox Aspect Ratio Distribution')
    plt.xlabel('Width/Height Ratio')
    plt.ylabel('Count')
    plt.yscale('log')
    
    # 类别分布
    cat_counts = {}
    for cat in cats:
        ann_ids = coco.getAnnIds(catIds=cat['id'])
        cat_counts[cat['name']] = len(ann_ids)
    
    plt.subplot(1, 3, 3)
    plt.bar(cat_counts.keys(), cat_counts.values())
    plt.title('Category Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'coco_statistics.png'), dpi=150)
    plt.close()
    
    # 可视化一些样本图片
    print(f"\nVisualizing {num_samples} samples...")
    
    # 随机选择带有标注的图片
    sample_img_ids = []
    for _ in range(min(num_samples * 3, len(img_ids))):  # 选取更多，以防有些图片加载失败
        img_id = random.choice(img_ids)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        if ann_ids:  # 只选择有标注的图片
            sample_img_ids.append(img_id)
        if len(sample_img_ids) >= num_samples:
            break
    
    # 使用不同颜色来表示不同类别
    cat_colors = plt.cm.rainbow(np.linspace(0, 1, len(cats)))
    cat_id_to_color = {cat['id']: cat_colors[i] for i, cat in enumerate(cats)}
    
    # 为每个样本创建可视化
    for i, img_id in enumerate(sample_img_ids[:num_samples]):
        img_info = coco.loadImgs(img_id)[0]
        
        # 构建正确的图片路径
        image_dir = os.path.dirname(os.path.dirname(coco.dataset["annotations_file"]))
        if "train" in coco.dataset["annotations_file"]:
            img_path = os.path.join(image_dir, "train2024", img_info['file_name'])
        else:
            img_path = os.path.join(image_dir, "val2024", img_info['file_name'])
        
        # 加载图片
        try:
            I = io.imread(img_path)
            
            # 创建图像
            plt.figure(figsize=(10, 10))
            plt.imshow(I)
            plt.axis('off')
            
            # 加载标注
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            # 绘制边界框
            for ann in anns:
                bbox = ann['bbox']  # [x, y, width, height]
                category_id = ann['category_id']
                
                # 获取类别名称和颜色
                cat_name = next((cat['name'] for cat in cats if cat['id'] == category_id), 'unknown')
                color = cat_id_to_color.get(category_id, (0, 0, 0, 1))
                
                # 绘制矩形
                rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                 linewidth=2, edgecolor=color, facecolor='none', alpha=0.7)
                plt.gca().add_patch(rect)
                
                # 添加类别标签
                # 如果有损伤信息，也显示
                if coco_info["has_damage_info"] and 'damage_type' in ann:
                    label = f"{cat_name} ({ann['damage_type']})"
                else:
                    label = cat_name
                    
                plt.text(bbox[0], bbox[1] - 5, label, color=color, 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0),
                         fontsize=10)
            
            plt.title(f'Sample {i+1}: {img_info["file_name"]} ({len(anns)} annotations)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{i+1}.png'), dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"Failed to visualize sample {i+1}: {str(e)}")
    
    print(f"Visualization results saved to: {output_dir}")

def main():
    # 设置路径
    coco_dir = "coco_spruce_beetle"
    annotations_dir = os.path.join(coco_dir, "annotations")
    train_json = os.path.join(annotations_dir, "instances_train2024.json")
    val_json = os.path.join(annotations_dir, "instances_val2024.json")
    
    # 检查文件是否存在
    if not os.path.exists(train_json):
        print(f"Training set annotation file does not exist: {train_json}")
        return
    
    if not os.path.exists(val_json):
        print(f"Validation set annotation file does not exist: {val_json}")
        return
    
    # 修复pycocotools的dataset路径问题
    for json_file in [train_json, val_json]:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 添加数据集路径信息
        data["annotations_file"] = json_file
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    
    # 验证训练集
    print("\n" + "="*50)
    print("VALIDATING TRAINING SET")
    print("="*50)
    train_info = validate_coco_dataset(train_json)
    if train_info:
        visualize_coco_dataset(train_info, os.path.join(coco_dir, "validation_train"), num_samples=5)
    
    # 验证测试集
    print("\n" + "="*50)
    print("VALIDATING VALIDATION SET")
    print("="*50)
    val_info = validate_coco_dataset(val_json)
    if val_info:
        visualize_coco_dataset(val_info, os.path.join(coco_dir, "validation_val"), num_samples=5)
    
    print("\nDataset validation completed")
    
    # 输出验证总结
    if train_info and val_info:
        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        print(f"Total images: {len(train_info['img_ids']) + len(val_info['img_ids'])}")
        print(f"Total annotations: {len(train_info['anns']) + len(val_info['anns'])}")
        print(f"Categories: {', '.join([cat['name'] for cat in train_info['cats']])}")
        print("\nDataset is valid and ready for training!")

if __name__ == "__main__":
    main()