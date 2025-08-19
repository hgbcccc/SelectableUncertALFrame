# tools/select_results.py
import os
import json
import argparse
import shutil
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='筛选主动学习结果')
    parser.add_argument('--work-dir', required=True, help='工作目录')
    parser.add_argument('--threshold', type=float, required=True, help='选择的阈值')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    return parser.parse_args()

def select_results(work_dir: Path, threshold: float, output_dir: Path):
    rounds = sorted(work_dir.glob('round_*'))
    
    for round_dir in rounds:
        teacher_outputs_dir = round_dir / 'teacher_outputs'
        if not teacher_outputs_dir.exists():
            print(f'警告: {teacher_outputs_dir} 不存在')
            continue
        
        # 查找时间相关的文件夹
        time_folders = sorted(teacher_outputs_dir.glob('*'))
        for time_folder in time_folders:
            results_dir = time_folder / 'results'
            visualize_dir = time_folder / 'visualize'  # 可视化文件夹
            if not results_dir.exists():
                print(f'警告: {results_dir} 不存在')
                continue
            
            for json_file in results_dir.glob('*.json'):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    scores = data.get('scores', [])
                    
                    if scores and any(score > threshold for score in scores):
                        # 提取文件名
                        base_name = json_file.stem.split('_result')[0]  # 获取基础名称
                        vis_image = visualize_dir / f"{base_name}_vis.jpg"  # 使用可视化文件夹

                        # 复制 JSON 文件和可视化图像
                        shutil.copy(json_file, output_dir / json_file.name)
                        if vis_image.exists():
                            shutil.copy(vis_image, output_dir / vis_image.name)
                            print(f'已选择: {json_file.name} 和 {vis_image.name}')
                        else:
                            print(f'警告: 可视化图像 {vis_image} 不存在')

def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    select_results(work_dir, args.threshold, output_dir)

if __name__ == '__main__':
    main()