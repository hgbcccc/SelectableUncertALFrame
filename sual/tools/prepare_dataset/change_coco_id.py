import json
from pathlib import Path


# 修改COCO格式标注文件中的类别名称和ID
def modify_categories(json_path: str, save_path: str = None):
    """修改COCO格式标注文件中的类别名称和ID
    
    Args:
        json_path: 输入标注文件路径
        save_path: 保存路径，如果为None则覆盖原文件
    """
    try:
        # 读取JSON文件
        print(f"读取文件: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 修改前的类别信息
        print("\n=== 修改前的类别 ===")
        for cat in data['categories']:
            print(f"类别ID: {cat['id']}, 名称: {cat['name']}")
            
        # 修改类别信息
        data['categories'] = [{'id': 0, 'name': 'canopy'}]  # 只保留一个类别
            
        # 修改标注中的类别ID
        for ann in data['annotations']:
            ann['category_id'] = 0  # 将所有类别ID改为0
            
        # 修改后的类别信息
        print("\n=== 修改后的类别 ===")
        for cat in data['categories']:
            print(f"类别ID: {cat['id']}, 名称: {cat['name']}")
            
        # 统计修改后的标注信息
        print(f"\n=== 标注统计 ===")
        print(f"总标注数量: {len(data['annotations'])}")
        category_counts = {}
        for ann in data['annotations']:
            cat_id = ann['category_id']
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        for cat_id, count in category_counts.items():
            print(f"类别ID {cat_id} 的标注数量: {count}")
            
        # 保存修改后的文件
        save_path = save_path or json_path
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"\n成功保存到: {save_path}")
        
        # 显示文件大小
        file_size = Path(save_path).stat().st_size / 1024 / 1024  # 转换为MB
        print(f"文件大小: {file_size:.2f}MB")
        
        return True
        
    except json.JSONDecodeError:
        print("错误: JSON格式无效")
        return False
    except Exception as e:
        print(f"错误: {str(e)}")
        return False

if __name__ == "__main__":
    # 输入和输出文件路径
    input_path = "data/Eucalyptus_canopy/annotations/instances_train2017.json"
    
    # 可以选择创建新文件或覆盖原文件
    # output_path = "data/Eucalyptus_canopy/annotations/instances_val2017_modified.json"  # 新文件
    output_path = None  # 覆盖原文件
    
    if Path(input_path).exists():
        modify_categories(input_path, output_path)
    else:
        print(f"错误: 文件不存在: {input_path}")