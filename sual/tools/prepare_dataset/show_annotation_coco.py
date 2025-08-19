import os
import cv2
import json
import argparse
from pycocotools.coco import COCO

def visualize_coco_annotations(json_file, image_dir, output_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载 COCO 数据
    coco = COCO(json_file)

    # 获取所有图像的 ID
    img_ids = coco.getImgIds()

    for img_id in img_ids:
        # 获取图像信息
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])
        image = cv2.imread(img_path)

        # 获取该图像的所有标注
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            # 获取边界框和类别
            bbox = ann['bbox']
            category_id = ann['category_id']
            category_name = coco.loadCats(category_id)[0]['name']

            # 画出边界框
            x, y, w, h = map(int, bbox)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 写上类别名称
            cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 保存图像
        output_path = os.path.join(output_dir, img_info['file_name'])
        cv2.imwrite(output_path, image)

def main():
    parser = argparse.ArgumentParser(description='Visualize COCO annotations.')
    parser.add_argument('--json_file', type=str, required=True, help='Path to the COCO JSON file.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save annotated images.')

    args = parser.parse_args()

    visualize_coco_annotations(args.json_file, args.image_dir, args.output_dir)

if __name__ == '__main__':
    main()