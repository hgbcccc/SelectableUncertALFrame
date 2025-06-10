import json
from pathlib import Path







#####
def verify_coco_format(json_path: str):
    """验证COCO格式标注文件
    
    Args:
        json_path: 标注文件路径
    """
    try:
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 检查必要的键
        required_keys = ['images', 'annotations', 'categories']
        for key in required_keys:
            if key not in data:
                print(f"错误: 缺少必要的键 '{key}'")
                return False
                
        # 检查images
        print("\n=== Images ===")
        print(f"图像数量: {len(data['images'])}")
        if data['images']:
            sample_image = data['images'][0]
            print("图像字段:", list(sample_image.keys()))
            print("示例图像:", sample_image)
            
        # 检查annotations
        print("\n=== Annotations ===")
        print(f"标注数量: {len(data['annotations'])}")
        if data['annotations']:
            sample_ann = data['annotations'][0]
            print("标注字段:", list(sample_ann.keys()))
            print("示例标注:", sample_ann)
            
        # 检查categories
        print("\n=== Categories ===")
        print(f"类别数量: {len(data['categories'])}")
        for cat in data['categories']:
            print(f"类别ID: {cat.get('id')}, 名称: {cat.get('name')}")
            
        # 验证关联
        image_ids = set(img['id'] for img in data['images'])
        ann_image_ids = set(ann['image_id'] for ann in data['annotations'])
        cat_ids = set(cat['id'] for cat in data['categories'])
        ann_cat_ids = set(ann['category_id'] for ann in data['annotations'])
        
        print("\n=== 验证结果 ===")
        print(f"图像ID数量: {len(image_ids)}")
        print(f"标注中的图像ID数量: {len(ann_image_ids)}")
        print(f"类别ID数量: {len(cat_ids)}")
        print(f"标注中的类别ID数量: {len(ann_cat_ids)}")
        
        # 检查ID对应关系
        orphan_anns = ann_image_ids - image_ids
        if orphan_anns:
            print(f"警告: 存在{len(orphan_anns)}个没有对应图像的标注")
            
        invalid_cats = ann_cat_ids - cat_ids
        if invalid_cats:
            print(f"警告: 存在{len(invalid_cats)}个无效的类别ID")
            
        return True
        
    except json.JSONDecodeError:
        print("错误: JSON格式无效")
        return False
    except Exception as e:
        print(f"错误: {str(e)}")
        return False

if __name__ == "__main__":
    # 验证文件
    json_path = "data/Eucalyptus_canopy/annotations/instances_train2017.json"
    print(f"验证文件: {json_path}")
    print("-" * 50)
    
    if Path(json_path).exists():
        verify_coco_format(json_path)
    else:
        print(f"错误: 文件不存在: {json_path}")