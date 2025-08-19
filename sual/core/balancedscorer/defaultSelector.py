from typing import Dict, List
import numpy as np
from datetime import datetime
import logging
'''
    Default selector, solves the problem that the default method does not generate a report.
'''

# 默认选择器，解决的是默认方法中没有生成报告的问题  
class DefaultSelector:
    def __init__(self, uncertainty_metric):
        self.uncertainty_metric = uncertainty_metric
        self.logger = logging.getLogger(__name__)

    
    # def select_samples(self, unlabeled_results: Dict, num_samples: int) -> List[str]:
    #     """基于不确定性度量选择样本"""
    #     uncertainty_scores = []
    #     valid_images = []
        
    #     for img_name, info in unlabeled_results.items():
    #         if 'uncertainty' not in info:
    #             continue
                
    #         uncertainty = info['uncertainty']
    #         if self.uncertainty_metric not in uncertainty:
    #             continue
                
    #         score = uncertainty[self.uncertainty_metric]
    #         if isinstance(score, dict):
    #             score = np.mean(list(score.values()))
                
    #         uncertainty_scores.append(score)
    #         valid_images.append(img_name)
        
    #     if not valid_images:
    #         return []
            
    #     # 排序选择
    #     uncertainty_scores = np.array(uncertainty_scores)
    #     num_samples = min(num_samples, len(valid_images))
    #     selected_indices = np.argsort(uncertainty_scores)[-num_samples:]  # 降序排序，选择最不确定的样本
        
    #     return [valid_images[i] for i in selected_indices]

    # def get_selection_report(self, selected_samples: List[str], unlabeled_results: Dict) -> Dict:
    #     """生成默认选择器的选择报告"""
    #     report = {
    #         'method': 'Default uncertainty-based selection',
    #         'metric': self.uncertainty_metric,
    #         'selected_samples': {},
    #         'statistics': {
    #             'total_selected': len(selected_samples),
    #             'uncertainty_scores': {}
    #         }
    #     }
        
    #     # 收集选中样本的不确定性分数
    #     uncertainty_scores = []
    #     for img_id in selected_samples:
    #         if img_id in unlabeled_results:
    #             score = unlabeled_results[img_id]['uncertainty'].get(self.uncertainty_metric, 0.0)
    #             report['selected_samples'][img_id] = {
    #                 'uncertainty_score': score
    #             }
    #             uncertainty_scores.append(score)
        
    #     # 计算统计信息
    #     if uncertainty_scores:
    #         report['statistics']['uncertainty_scores'] = {
    #             'mean': float(np.mean(uncertainty_scores)),
    #             'std': float(np.std(uncertainty_scores)),
    #             'min': float(np.min(uncertainty_scores)),
    #             'max': float(np.max(uncertainty_scores)),
    #             'median': float(np.median(uncertainty_scores))
    #         }
        
    #     return report

                
    def select_samples(self, unlabeled_results: Dict, num_samples: int,train_stats=None) -> List[str]:
        """基于不确定性度量选择样本"""
        try:
            uncertainty_scores = []
            valid_images = []
            self.score_details = {}  # 存储详细信息供报告使用
            
            # 处理每个样本
            for img_name, info in unlabeled_results.items():
                if 'uncertainty' not in info:
                    self.logger.warning(f"图片 {img_name} 缺少不确定性信息")
                    continue
                
                uncertainty = info['uncertainty']
                if self.uncertainty_metric not in uncertainty:
                    self.logger.warning(
                        f"图片 {img_name} 缺少 {self.uncertainty_metric} 不确定性度量")
                    continue
                
                score = uncertainty[self.uncertainty_metric]
                if isinstance(score, dict):
                    score = np.mean(list(score.values()))
                
                # 记录完整的不确定性信息
                self.score_details[img_name] = {
                    'score': float(score),
                    'rank': None,  # 稍后填充
                    'selected': False,  # 是否被选中
                    'all_uncertainty_metrics': {
                        k: float(v) if isinstance(v, (int, float, np.number)) 
                        else v for k, v in uncertainty.items()
                    },
                    'predictions': {
                        'num_predictions': len(info.get('predictions', [])),
                        'prediction_scores': [
                            float(p['score']) for p in info.get('predictions', [])
                        ] if 'predictions' in info else [],
                        'prediction_labels': [
                            p['label'] for p in info.get('predictions', [])
                        ] if 'predictions' in info else []
                    }
                }
                
                uncertainty_scores.append(score)
                valid_images.append(img_name)

            if not valid_images:
                self.logger.error("没有找到有效的样本")
                return []

            # 选择不确定性最高的样本
            uncertainty_scores = np.array(uncertainty_scores)
            num_samples = min(num_samples, len(valid_images))
            
            # 获取完整的排序索引
            full_ranking = np.argsort(uncertainty_scores)[::-1]  # 降序排序
            
            # 更新每个样本的排名信息
            self.ranked_results = []  # 存储按排名排序的结果
            for rank, idx in enumerate(full_ranking, 1):
                img_name = valid_images[idx]
                self.score_details[img_name]['rank'] = rank
                self.score_details[img_name]['selected'] = rank <= num_samples
                
                # 添加到排序结果列表
                self.ranked_results.append({
                    'rank': rank,
                    'image_name': img_name,
                    'score': float(uncertainty_scores[idx]),
                    'selected': rank <= num_samples,
                    **self.score_details[img_name]  # 包含所有详细信息
                })
            
            # 选择top-k样本
            selected_indices = full_ranking[:num_samples]
            selected = [valid_images[i] for i in selected_indices]
            
            # 保存统计信息供报告使用
            self.statistics = {
                'mean_score': float(np.mean(uncertainty_scores)),
                'std_score': float(np.std(uncertainty_scores)),
                'min_score': float(np.min(uncertainty_scores)),
                'max_score': float(np.max(uncertainty_scores)),
                'median_score': float(np.median(uncertainty_scores)),
                'quartiles': {
                    'q25': float(np.percentile(uncertainty_scores, 25)),
                    'q75': float(np.percentile(uncertainty_scores, 75))
                },
                'selected_stats': {
                    'mean_score': float(np.mean(uncertainty_scores[selected_indices])),
                    'std_score': float(np.std(uncertainty_scores[selected_indices]))
                }
            }
            
            return selected
            
        except Exception as e:
            self.logger.error(f"样本选择失败: {str(e)}")
            return []

    def get_selection_report(self, selected_samples: List[str], unlabeled_results: Dict) -> Dict:
        """生成选择报告"""
        ranking_info = {
            'metadata': {
                'metric': self.uncertainty_metric,
                'total_samples': len(self.score_details),
                'selected_samples': len(selected_samples),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            'statistics': self.statistics,
            'detailed_results': {
                'all_samples': self.score_details,  # 所有样本的详细信息
                'ranked_list': self.ranked_results,  # 按排名排序的结果
                'selected_samples': selected_samples  # 最终选择的样本
            }
        }
        
        # 打印选择的样本信息
        self.logger.info(f"\n选择了 {len(selected_samples)} 个样本:")
        self.logger.info("\n排序详细信息:")
        self.logger.info(f"{'排名':<6}{'图片名称':<50}{'分数':<10}{'是否选中':<10}")
        self.logger.info("-" * 76)
        
        # 打印前20个样本的信息（包括选中和未选中的）
        for result in self.ranked_results[:20]:
            self.logger.info(
                f"{result['rank']:<6}{result['image_name']:<50}"
                f"{result['score']:.4f}  {'√' if result['selected'] else '×'}"
            )
        
        return ranking_info