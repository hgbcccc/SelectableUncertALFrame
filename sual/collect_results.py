# collect_results.py

import pandas as pd
import os
from pathlib import Path
import numpy as np

def collect_experiment_results():
    # 实验名称映射
    experiment_labels = {
        # 'al_faster-rcnn_ssc_combinatorial': 'Combinatorial',
        # 'al_faster-rcnn_ssc_default': 'Default',
        # 'al_faster-rcnn_ssc_rl_kl': 'RL-KL',
        # 'al_faster-rcnn_ssc_wasserstein': 'Wasserstein',
        # "al_faster-rcnn_ssc_rl_wasserstein": "RL-Wasserstein",
        # "faster-rcnn_entropy": "Entropy",
        # "faster-rcnn_least_confid": "Least Confid",
        # "faster-rcnn_sor": "SOR"

        "faster-rcnn_ssc_combinatorial_16_200": "Combinatorial_16_200",
        "faster-rcnn_ssc_default_16_200": "Default_16_200",
        
    }
    
    # 要提取的轮次
    target_rounds = [4, 8, 12, 16]
    
    # 要提取的指标
    metrics = ['val_bbox_mAP', 'val_bbox_mAP_50', 'val_bbox_mAP_75']
    
    # 存储所有结果
    all_results = []
    
    # 遍历work_dirs下的所有目录
    work_dirs = os.listdir("work_dirs")
    for work_dir in work_dirs:
        if work_dir.startswith("al_faster-rcnn_ssc_") or work_dir.startswith("faster-rcnn_"):
            # 获取实验的显示名称
            display_name = experiment_labels.get(work_dir, work_dir)
            
            # 读取performance_history.csv
            csv_path = os.path.join("work_dirs", work_dir, "performance_history.csv")
            if not os.path.exists(csv_path):
                print(f"警告: {csv_path} 不存在")
                continue
                
            try:
                df = pd.read_csv(csv_path)
                
                # 确保'round'列存在
                if 'round' not in df.columns:
                    print(f"警告: {work_dir} 的CSV文件中没有'round'列")
                    continue
                
                # 提取目标轮次的结果
                round_results = {}
                for round_num in target_rounds:
                    round_data = df[df['round'] == round_num]
                    if len(round_data) == 0:
                        print(f"警告: {work_dir} 中没有找到轮次 {round_num} 的结果")
                        continue
                        
                    # 获取该轮次的指标值
                    for metric in metrics:
                        if metric in round_data.columns:
                            col_name = f"R{round_num}_{metric}"
                            round_results[col_name] = round_data[metric].iloc[0]
                        else:
                            print(f"警告: {work_dir} 中没有找到指标 {metric}")
                
                # 添加到结果列表
                if round_results:
                    round_results['Method'] = display_name
                    all_results.append(round_results)
                    
            except Exception as e:
                print(f"处理 {work_dir} 时出错: {str(e)}")
    
    # 创建结果DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # 设置Method列为索引
        results_df.set_index('Method', inplace=True)
        
        # 对列进行排序，确保顺序一致
        cols = []
        for round_num in target_rounds:
            for metric in metrics:
                cols.append(f"R{round_num}_{metric}")
        results_df = results_df.reindex(columns=cols)
        
        # 保存结果
        output_file = "experiment_results_summary.csv"
        results_df.to_csv(output_file)
        print(f"\n结果已保存到: {output_file}")
        
        # 打印结果表格
        print("\n实验结果汇总:")
        print(results_df.round(4))
        
        # 计算每个方法在最后一轮的性能排名
        final_round = target_rounds[-1]
        final_map = f"R{final_round}_val_bbox_mAP"
        if final_map in results_df.columns:
            print(f"\n第{final_round}轮 mAP 排名:")
            rankings = results_df[final_map].sort_values(ascending=False)
            for i, (method, score) in enumerate(rankings.items(), 1):
                print(f"{i}. {method}: {score:.4f}")
    else:
        print("没有找到有效的实验结果")

if __name__ == "__main__":
    collect_experiment_results()