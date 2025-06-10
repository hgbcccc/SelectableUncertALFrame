import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import argparse
import os
from matplotlib import font_manager

# 设置中文字体
def set_chinese_font():
    """设置中文字体"""
    try:
        # Windows系统
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    except:
        try:
            # Linux系统
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        except:
            print("警告: 未能找到合适的中文字体，可能会导致中文显示异常")
    plt.rcParams['axes.unicode_minus'] = False


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

def analyze_detection_results(npz_path, meta_path, output_dir=None):
    """分析检测结果"""
    # 设置中文字体
    set_chinese_font()
    
    print(f"\n=== 分析检测结果 ===")
    print(f"NPZ文件: {npz_path}")
    print(f"Meta文件: {meta_path}")
    
    # 1. 加载数据
    try:
        data = np.load(npz_path, allow_pickle=True)
        with open(meta_path, 'r') as f:
            meta_info = json.load(f)
    except Exception as e:
        print(f"错误: 无法加载文件 - {str(e)}")
        return
    
    # 2. 分析FPN特征
    print("\n=== FPN特征分析 ===")
    fpn_keys = [k for k in data.keys() if k.startswith('fpn_')]
    for key in fpn_keys:
        feat = data[key]
        print(f"\n{key}:")
        print(f"形状: {feat.shape}")
        print(f"均值: {feat.mean():.4f}")
        print(f"标准差: {feat.std():.4f}")
        print(f"最大值: {feat.max():.4f}")
        print(f"最小值: {feat.min():.4f}")
        print(f"激活通道数: {np.sum(feat.mean(axis=(2,3)) > 0)}")
    
    # 3. 分析分类分数
    print("\n=== 分类分数分析 ===")
    cls_keys = [k for k in data.keys() if k.startswith('cls_scores_')]
    all_scores = []
    for key in cls_keys:
        scores = data[key]
        print(f"\n{key}:")
        print(f"形状: {scores.shape}")
        print(f"分数范围: {scores.min():.4f} - {scores.max():.4f}")
        print(f"平均分数: {scores.mean():.4f}")
        all_scores.append(scores)
        
        # 分析每个特征层的最高分数
        max_scores_per_class = scores[0].max(axis=(1,2))  # [80]
        top_classes = np.argsort(max_scores_per_class)[-5:]
        print(f"\n该特征层置信度最高的5个类别:")
        for class_idx in reversed(top_classes):
            print(f"类别 {class_idx} ({get_coco_classes()[class_idx]}): "
                  f"最高分数 = {max_scores_per_class[class_idx]:.4f}")
    
    # 4. 分析边界框预测
    print("\n=== 边界框预测分析 ===")
    bbox_keys = [k for k in data.keys() if k.startswith('bbox_preds_')]
    for key in bbox_keys:
        preds = data[key]
        print(f"\n{key}:")
        print(f"形状: {preds.shape}")
        print(f"均值: {preds.mean(axis=(2,3))}")
        print(f"标准差: {preds.std(axis=(2,3))}")
    
    # 5. 可视化
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 5.1 FPN特征可视化
        for key in fpn_keys:
            feat = data[key]
            plt.figure(figsize=(15, 5))
            
            # 特征图平均激活
            plt.subplot(131)
            mean_activation = feat[0].mean(axis=0)
            plt.imshow(mean_activation, cmap='viridis')
            plt.colorbar()
            plt.title(f'{key}\n平均激活')
            
            # 特征图最大激活
            plt.subplot(132)
            max_activation = feat[0].max(axis=0)
            plt.imshow(max_activation, cmap='viridis')
            plt.colorbar()
            plt.title('最大激活')
            
            # 通道激活分布
            plt.subplot(133)
            channel_means = feat[0].mean(axis=(1,2))
            plt.hist(channel_means, bins=30, edgecolor='black')
            plt.title('通道激活分布')
            plt.xlabel('平均激活值')
            plt.ylabel('通道数量')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{key}_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5.2 分类分数可视化
        plt.figure(figsize=(15, 5))
        for i, (key, scores) in enumerate(zip(cls_keys, all_scores)):
            plt.subplot(1, len(cls_keys), i+1)
            plt.hist(scores.ravel(), bins=50, edgecolor='black')
            plt.title(f'{key}\n分数分布')
            plt.xlabel('分数')
            plt.ylabel('数量')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cls_scores_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n分析图表已保存至: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='分析目标检测预测结果')
    parser.add_argument('npz_path', help='预测结果NPZ文件路径')
    parser.add_argument('meta_path', help='预测结果Meta文件路径')
    parser.add_argument('--output-dir', '-o', help='输出目录（可选）')
    
    args = parser.parse_args()
    analyze_detection_results(args.npz_path, args.meta_path, args.output_dir)

if __name__ == "__main__":
    main()
    
    
# !python tools/analyze_origin_det.py 