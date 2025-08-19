import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import time

# 验证JSON文件的有效性和完整性
#验证
class JsonVerifier:
    """JSON文件验证器"""
    
    @staticmethod
    def verify_json_file(json_path: str) -> Optional[Dict]:
        """验证JSON文件的有效性和完整性
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            Dict: 包含验证结果的字典，如果验证失败则返回None
        """
        start_time = time.time()
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 检查是否是COCO格式
            if not isinstance(data, dict):
                print(f"错误: {json_path} 不是有效的JSON对象")
                return None
                
            # 基本结构验证
            required_keys = {'images', 'annotations', 'categories'}
            if not all(key in data for key in required_keys):
                missing_keys = required_keys - set(data.keys())
                print(f"错误: 缺少必需的键 {missing_keys}")
                return None
                
            # 统计信息
            stats = {
                'file_name': Path(json_path).name,
                'file_size_mb': Path(json_path).stat().st_size / (1024 * 1024),
                'num_images': len(data['images']),
                'num_annotations': len(data['annotations']),
                'num_categories': len(data['categories']),
                'processing_time': time.time() - start_time
            }
            
            # 验证图像ID的唯一性
            image_ids = {img['id'] for img in data['images']}
            if len(image_ids) != len(data['images']):
                print(f"警告: 存在重复的图像ID")
                
            # 验证标注的图像ID是否存在
            invalid_ann_count = 0
            for ann in data['annotations']:
                if ann['image_id'] not in image_ids:
                    invalid_ann_count += 1
                    
            if invalid_ann_count > 0:
                print(f"警告: 发现 {invalid_ann_count} 个标注引用了不存在的图像ID")
                
            # 计算每个类别的标注数量
            category_counts = {}
            for ann in data['annotations']:
                cat_id = ann['category_id']
                category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
                
            stats['category_distribution'] = category_counts
            
            return stats
            
        except json.JSONDecodeError as e:
            print(f"错误: JSON解析失败 - {str(e)}")
            return None
        except Exception as e:
            print(f"错误: 验证失败 - {str(e)}")
            return None
            
    @staticmethod
    def print_stats(stats: Dict):
        """打印统计信息"""
        print(f"\n验证文件: {stats['file_name']}")
        print(f"文件大小: {stats['file_size_mb']:.2f} MB")
        print(f"处理时间: {stats['processing_time']:.2f} 秒")
        print("\n数据统计:")
        print(f"图像数量: {stats['num_images']}")
        print(f"标注数量: {stats['num_annotations']}")
        print(f"类别数量: {stats['num_categories']}")
        
        print("\n每个类别的标注数量:")
        for cat_id, count in stats['category_distribution'].items():
            print(f"类别 {cat_id}: {count}")
            
def verify_json_folder(folder_path: str):
    """验证文件夹中的所有JSON文件
    
    Args:
        folder_path: 文件夹路径
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"错误: 文件夹 {folder_path} 不存在")
        return False
        
    # 获取所有JSON文件
    json_files = list(folder.glob('*.json'))
    if not json_files:
        print(f"错误: 在 {folder_path} 中没有找到JSON文件")
        return False
        
    print(f"找到 {len(json_files)} 个JSON文件:")
    for json_file in json_files:
        print(f"  - {json_file.name}")
        
    verifier = JsonVerifier()
    all_valid = True
    all_stats = []
    
    for json_file in json_files:
        print(f"\n验证 {json_file.name}...")
        stats = verifier.verify_json_file(str(json_file))
        
        if stats is None:
            all_valid = False
            continue
            
        verifier.print_stats(stats)
        all_stats.append(stats)
        
    if all_valid and all_stats:
        print("\n📊 数据集总览:")
        total_images = sum(s['num_images'] for s in all_stats)
        total_annotations = sum(s['num_annotations'] for s in all_stats)
        print(f"总图像数量: {total_images}")
        print(f"总标注数量: {total_annotations}")
        print(f"平均每张图片的标注数量: {total_annotations/total_images:.2f}")
        
    return all_valid

def parse_args():
    parser = argparse.ArgumentParser(description='验证JSON文件')
    parser.add_argument('folder_path', help='包含JSON文件的文件夹路径')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    is_valid = verify_json_folder(args.folder_path)
    
    if is_valid:
        print("\n✅ 验证通过: 所有文件都是有效的")
    else:
        print("\n❌ 验证失败: 请检查上述错误")