# # tools/analyze_al_result.py
# import os
# import json
# import argparse
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path

# def parse_args():
#     parser = argparse.ArgumentParser(description='分析主动学习训练结果')
#     parser.add_argument('--work-dir', required=True, help='工作目录')
#     return parser.parse_args()

# def analyze_results(work_dir: Path):
#     iterations = sorted(work_dir.glob('iteration_*'))
    
#     for iteration in iterations:
#         teacher_outputs_dir = iteration / 'teacher_outputs'
#         if not teacher_outputs_dir.exists():
#             print(f'警告: {teacher_outputs_dir} 不存在')
#             continue
        
#         # 查找时间相关的文件夹
#         time_folders = sorted(teacher_outputs_dir.glob('*'))
#         for time_folder in time_folders:
#             results_dir = time_folder / 'results'
#             if not results_dir.exists():
#                 print(f'警告: {results_dir} 不存在')
#                 continue
            
#             scores = []
#             for json_file in results_dir.glob('*.json'):
#                 with open(json_file, 'r') as f:
#                     data = json.load(f)
#                     scores.extend(data.get('scores', []))
            
#             if scores:
#                 plt.figure(figsize=(10, 6))
#                 sns.histplot(scores, bins=30, kde=True, color='blue', stat='density', alpha=0.6)
#                 iteration_number = iteration.name.split('_')[-1]  # 获取迭代次数
#                 # iteration_number = int(iteration_number)  # 训练初始次数为0 
#                 # iteration_number = iteration_number+1
#                 plt.title(f'Scores Distribution of Active Learning Iteration {iteration_number}', fontsize=16)
#                 plt.xlabel('Scores', fontsize=14)
#                 plt.ylabel('Density', fontsize=14)
#                 plt.grid(True)
#                 plt.tight_layout()
#                 plt.savefig(work_dir / f'{iteration.name}_scores_distribution.png')
#                 plt.close()
#                 print(f'已保存 {iteration.name} 的分数分布直方图')
#             else:
#                 print(f'警告: {time_folder.name} 中未找到分数数据')

# def main():
#     args = parse_args()
#     work_dir = Path(args.work_dir)
#     analyze_results(work_dir)

# if __name__ == '__main__':
#     main()

# tools/analyze_al_result.py


########################################## 分析主动学习训练结果 ########################################## 
########################################## 输出各个轮次中分数分布直方图 ########################################## 


import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='分析主动学习训练结果')
    parser.add_argument('--work-dir', required=True, help='工作目录')
    return parser.parse_args()

def analyze_results(work_dir: Path):
    rounds = sorted(work_dir.glob('round_*'))  # 修改为 round
    
    for round_dir in rounds:
        teacher_outputs_dir = round_dir / 'teacher_outputs'
        if not teacher_outputs_dir.exists():
            print(f'警告: {teacher_outputs_dir} 不存在')
            continue
        
        # 查找时间相关的文件夹
        time_folders = sorted(teacher_outputs_dir.glob('*'))
        for time_folder in time_folders:
            results_dir = time_folder / 'results'
            if not results_dir.exists():
                print(f'警告: {results_dir} 不存在')
                continue
            
            scores = []
            for json_file in results_dir.glob('*.json'):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    scores.extend(data.get('scores', []))
            
            if scores:
                plt.figure(figsize=(10, 6))
                sns.histplot(scores, bins=30, kde=True, color='blue', stat='density', alpha=0.6)
                round_number = round_dir.name.split('_')[-1]  # 获取轮次
                round_number = int(round_number)
                round_number= round_number+1
                plt.title(f'Scores Distribution of Active Learning Round {round_number}', fontsize=16)
                plt.xlabel('Scores', fontsize=14)
                plt.ylabel('Density', fontsize=14)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(work_dir / f'{round_dir.name}_scores_distribution.png')
                plt.close()
                print(f'已保存 {round_dir.name} 的分数分布直方图')
            else:
                print(f'警告: {time_folder.name} 中未找到分数数据')

def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    analyze_results(work_dir)

if __name__ == '__main__':
    main()