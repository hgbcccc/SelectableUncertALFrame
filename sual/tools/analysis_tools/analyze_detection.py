import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import argparse
import os









#############################################################################################
def set_matplotlib_chinese():
    """设置matplotlib支持中文显示"""
    try:
        # Windows系统优先使用微软雅黑
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("成功设置中文字体")
    except Exception as e:
        print(f"警告：设置中文字体时出现问题 - {str(e)}")
        print("图表中的中文可能无法正确显示")

def get_coco_classes():
    """获取完整的COCO数据集类别映射"""
    return {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
        35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
        39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
        44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
        49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
        54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
        59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
        64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
        69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
        74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
        79: 'toothbrush'
    }

def analyze_detection_json(json_path, output_dir=None):
    """分析检测结果JSON文件并输出总结信息"""
    
    # 设置中文字体
    set_matplotlib_chinese()
    
    # 获取COCO类别映射
    COCO_CLASSES = get_coco_classes()
    
    # 读取JSON文件
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：{json_path} 不是有效的JSON文件")
        return
    
    labels = np.array(data['labels'])
    scores = np.array(data['scores'])
    bboxes = np.array(data['bboxes'])
    
    # 1. 基本统计信息
    print("\n=== 基本检测统计 ===")
    print(f"总检测框数量: {len(labels)}")
    print(f"置信度范围: {scores.min():.3f} - {scores.max():.3f}")
    print(f"平均置信度: {scores.mean():.3f}")
    
    # 2. 类别分布
    print("\n=== 类别分布 (Top 5) ===")
    label_counts = Counter(labels)
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        class_name = COCO_CLASSES.get(label, f'class_{label}')
        print(f"{class_name}: {count} 个 ({count/len(labels)*100:.1f}%)")
    
    # 3. 置信度分布
    print("\n=== 置信度分布 ===")
    confidence_ranges = {
        '0.8-1.0': len(scores[scores >= 0.8]),
        '0.6-0.8': len(scores[(scores >= 0.6) & (scores < 0.8)]),
        '0.4-0.6': len(scores[(scores >= 0.4) & (scores < 0.6)]),
        '0.2-0.4': len(scores[(scores >= 0.2) & (scores < 0.4)]),
        '0.0-0.2': len(scores[scores < 0.2])
    }
    for range_name, count in confidence_ranges.items():
        print(f"{range_name}: {count} 个 ({count/len(scores)*100:.1f}%)")
    
    # 4. 边界框分析
    print("\n=== 边界框分析 ===")
    box_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    print(f"最小框面积: {box_areas.min():.1f}")
    print(f"最大框面积: {box_areas.max():.1f}")
    print(f"平均框面积: {box_areas.mean():.1f}")
    
    # 5. 高置信度检测（>0.5）的类别分布
    high_conf_mask = scores > 0.5
    high_conf_labels = labels[high_conf_mask]
    if len(high_conf_labels) > 0:
        print("\n=== 高置信度检测(>0.5)的类别分布 ===")
        high_conf_counts = Counter(high_conf_labels)
        for label, count in sorted(high_conf_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            class_name = COCO_CLASSES.get(label, f'class_{label}')
            print(f"{class_name}: {count} 个")
    
    # 6. 可视化
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 6.1 置信度分布直方图
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, edgecolor='black')
        plt.title('检测置信度分布', fontsize=12)
        plt.xlabel('置信度', fontsize=10)
        plt.ylabel('检测框数量', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        confidence_plot_path = os.path.join(output_dir, 'confidence_distribution.png')
        plt.savefig(confidence_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n置信度分布图已保存至: {confidence_plot_path}")
        
        # 6.2 类别分布条形图
        plt.figure(figsize=(12, 6))
        top_classes = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        class_names = [COCO_CLASSES.get(label, f'class_{label}') for label, _ in top_classes]
        counts = [count for _, count in top_classes]
        
        plt.bar(range(len(counts)), counts)
        plt.xticks(range(len(counts)), class_names, rotation=45, ha='right')
        plt.title('Top 10 检测类别分布', fontsize=12)
        plt.xlabel('类别', fontsize=10)
        plt.ylabel('检测框数量', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        classes_plot_path = os.path.join(output_dir, 'class_distribution.png')
        plt.savefig(classes_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"类别分布图已保存至: {classes_plot_path}")

def main():
    parser = argparse.ArgumentParser(description='分析目标检测结果JSON文件')
    parser.add_argument('json_path', help='检测结果JSON文件的路径')
    parser.add_argument('--output-dir', '-o', help='输出目录路径（可选）')
    
    args = parser.parse_args()
    analyze_detection_json(args.json_path, args.output_dir)

if __name__ == "__main__":
    main()
    
    
    
    
    
# python analyze_detection.py demo.json -o outputs/analysis