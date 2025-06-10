from typing import Dict, List
import numpy as np


'''
    Convert unlabeled pool results to stats format utility functions.

    This module contains helper functions for converting unlabeled pool results to stats format, including:
    - Converting unlabeled pool results to stats format
'''
def convert_unlabeled_to_stats_format(unlabeled_pool_results: Dict) -> Dict:
    """
    将未标注池数据转换为与train_stats相似的格式
    
    Args:
        unlabeled_pool_results: 未标注池数据
        {
            'img_id.jpg': {
                'uncertainty': {
                    'ssc_score': float,
                    'occlusion_score': float,
                    'crown_count_score': float,
                    'diversity_score': float,
                    'area_var_score': float,
                    'density_var_score': float
                }
            },
            ...
        }
    
    Returns:
        {
            'features': {
                'ssc_score': {
                    'mean': float,
                    'std': float,
                    'distribution': List[float],
                    'min': float,
                    'max': float
                },
                'occlusion_score': {
                    'mean': float,
                    'std': float,
                    'distribution': List[float],
                    'min': float,
                    'max': float
                },
                # 其他特征...
            },
            'image_mapping': {
                'image_names': List[str],
                'indices': Dict[str, int]
            }
        }
    """
    # 初始化特征列表 - 确保与输入数据结构匹配
    features = [
        'ssc_score',
        'occlusion_score',
        'crown_count_score',
        'diversity_score',
        'area_var_score',
        'density_var_score'
    ]
    
    # 初始化返回的数据结构
    stats = {
        'features': {
            feature: {
                'distribution': [],
                'mean': 0.0,
                'std': 0.0,
                'min': float('inf'),
                'max': float('-inf')
            } for feature in features
        },
        'image_mapping': {
            'image_names': [],
            'indices': {}
        }
    }
    
    # 收集数据
    for idx, (image_name, data) in enumerate(unlabeled_pool_results.items()):
        # 保存图片名称和索引映射
        stats['image_mapping']['image_names'].append(image_name)
        stats['image_mapping']['indices'][image_name] = idx
        
        # 收集每个特征的值 - 直接从uncertainty中获取
        uncertainty = data['uncertainty']
        for feature in features:
            # 确保特征存在于uncertainty字典中
            if feature in uncertainty:
                value = uncertainty[feature]
                stats['features'][feature]['distribution'].append(value)
                
                # 更新最大最小值
                stats['features'][feature]['min'] = min(
                    stats['features'][feature]['min'], 
                    value
                )
                stats['features'][feature]['max'] = max(
                    stats['features'][feature]['max'], 
                    value
                )
            else:
                # 如果特征不存在，记录警告
                import warnings
                warnings.warn(f"特征 {feature} 未在图片 {image_name} 的uncertainty中找到")
    
    # 计算统计信息
    for feature in features:
        if stats['features'][feature]['distribution']:  # 确保分布不为空
            values = np.array(stats['features'][feature]['distribution'])
            stats['features'][feature]['mean'] = float(np.mean(values))
            stats['features'][feature]['std'] = float(np.std(values))
        else:
            # 如果分布为空，设置默认值
            stats['features'][feature]['mean'] = 0.0
            stats['features'][feature]['std'] = 0.0
            stats['features'][feature]['min'] = 0.0
            stats['features'][feature]['max'] = 0.0
    
    return stats