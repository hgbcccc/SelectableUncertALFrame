
import os
import json
import cv2
import numpy as np
import argparse
from pathlib import Path


def load_json(json_path):
    """加载 JSON 文件"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {json_path}. {e}")
        return None


def calculate_iou(box1, box2):
    """
    计算两个边界框的 IoU
    :param box1: [x, y, w, h]
    :param box2: [x, y, w, h]
    :return: IoU 值
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area


def generate_colors(num_classes):
    """
    生成随机颜色，每个类别一个颜色
    :param num_classes: 类别总数
    :return: 类别 ID 到颜色的映射字典
    """
    np.random.seed(42)  # 固定随机种子
    colors = [tuple(np.random.randint(0, 255, size=3).tolist()) for _ in range(num_classes)]
    return {i: color for i, color in enumerate(colors)}


def draw_bboxes_with_labels(image, bbox_pairs, output_path, class_colors):
    """
    在图片上绘制多个边界框以及标签
    :param image: 输入图片
    :param bbox_pairs: [(box1, box2, label1, label2), ...]，包含多个边界框对和对应的标签
    :param output_path: 保存绘制好的图片路径
    :param class_colors: 类别 ID 到颜色的映射
    """
    for box1, box2, label1, label2 in bbox_pairs:
        # 绘制第一个框
        x1, y1, w1, h1 = map(int, box1)
        color1 = class_colors.get(label1, (0, 255, 0))  # 如果类别未定义颜色，使用默认绿色
        cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), color1, 2)
        label1_text = f"Class {label1}"
        label1_size = cv2.getTextSize(label1_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(image, (x1, y1 - label1_size[1] - 4), (x1 + label1_size[0] + 4, y1), color1, -1)
        cv2.putText(image, label1_text, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # 绘制第二个框
        x2, y2, w2, h2 = map(int, box2)
        color2 = class_colors.get(label2, (255, 0, 0))  # 如果类别未定义颜色，使用默认蓝色
        cv2.rectangle(image, (x2, y2), (x2 + w2, y2 + h2), color2, 2)
        label2_text = f"Class {label2}"
        label2_size = cv2.getTextSize(label2_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(image, (x2, y2 - label2_size[1] - 4), (x2 + label2_size[0] + 4, y2), color2, -1)
        cv2.putText(image, label2_text, (x2 + 2, y2 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imwrite(output_path, image)


def find_coco_dataset_paths(base_path):
    """
    自动找到 COCO 数据集的路径，包括 annotations 和 train2017/val2017
    """
    annotations_path = os.path.join(base_path, 'annotations')
    train_images_path = os.path.join(base_path, 'train2017')
    val_images_path = os.path.join(base_path, 'val2017')

    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"'annotations' directory not found in {base_path}")
    if not os.path.exists(train_images_path):
        raise FileNotFoundError(f"'train2017' directory not found in {base_path}")
    if not os.path.exists(val_images_path):
        raise FileNotFoundError(f"'val2017' directory not found in {base_path}")

    annotation_files = []
    for filename in ['instances_train2017.json', 'instances_val2017.json']:
        file_path = os.path.join(annotations_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Annotation file {filename} not found in {annotations_path}")
        annotation_files.append(file_path)

    return {
        "annotations": annotation_files,
        "train_images": train_images_path,
        "val_images": val_images_path,
    }


def process_coco_dataset(args):
    """
    根据 IoU 筛选边界框并将结果保存
    """
    dataset_paths = find_coco_dataset_paths(args.dataset_path)
    annotation_files = dataset_paths['annotations']
    images_paths = [dataset_paths['train_images'], dataset_paths['val_images']]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成类别颜色映射
    num_classes = 80  # COCO 数据集类别数量
    class_colors = generate_colors(num_classes)

    iou_ranges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    iou_folders = [output_dir / f"IOU{int(iou * 100)}" for iou in iou_ranges[:-1]]
    for folder in iou_folders:
        folder.mkdir(exist_ok=True)

    for annotation_file, image_dir in zip(annotation_files, images_paths):
        coco_data = load_json(annotation_file)
        if coco_data is None:
            continue

        images = {img['id']: img for img in coco_data['images']}
        annotations = coco_data['annotations']

        # 遍历所有图片
        for image_id, image_info in images.items():
            image_path = os.path.join(image_dir, image_info['file_name'])
            image_annotations = [(ann['bbox'], ann.get('category_id', 'unknown')) for ann in annotations if ann['image_id'] == image_id]

            if len(image_annotations) < 2:  # 至少需要两个框才能计算 IoU
                continue

            # 加载图片
            image = cv2.imread(image_path)
            if image is None:
                print(f"Cannot read image: {image_path}")
                continue

            # 存储每个 IoU 范围的框对
            iou_bbox_pairs = [[] for _ in range(len(iou_ranges) - 1)]

            # 计算 IoU
            for i, (box1, label1) in enumerate(image_annotations):
                for (box2, label2) in image_annotations[i + 1:]:
                    iou = calculate_iou(box1, box2)
                    for j in range(len(iou_ranges) - 1):
                        if iou_ranges[j] <= iou < iou_ranges[j + 1]:
                            iou_bbox_pairs[j].append((box1, box2, label1, label2))

            # 保存图片
            for j, bbox_pairs in enumerate(iou_bbox_pairs):
                if bbox_pairs:
                    output_image_path = os.path.join(iou_folders[j], image_info['file_name'])
                    draw_bboxes_with_labels(image.copy(), bbox_pairs, output_image_path, class_colors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO Dataset IoU Filter and Visualization")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to COCO dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    args = parser.parse_args()

    process_coco_dataset(args)
