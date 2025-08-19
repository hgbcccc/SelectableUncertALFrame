import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
# 在 Jupyter Notebook 中显示图形


def traverse_and_extract_scores(folder_path):
    all_scores = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if 'scores' in data:
                            scores = data['scores']
                            if isinstance(scores, list):
                                all_scores.extend(scores)
                            elif isinstance(scores, (int, float)):
                                all_scores.append(scores)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return all_scores

def plot_histogram(scores):
    sns.set_style("whitegrid")  # 设置绘图风格为白色网格背景
    fig, ax = plt.subplots(figsize=(8, 6))  # 设置图形大小
    sns.histplot(scores, bins=30, kde=True, ax=ax)  # 使用seaborn的histplot，添加核密度估计（kde）
    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('The Confidence Distribution of Teacher Model Inference', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)  # 设置刻度字体大小
    fig.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Extract and plot scores from JSON files.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing JSON files.')
    args = parser.parse_args()

    scores = traverse_and_extract_scores(args.folder_path)
    if scores:
        plot_histogram(scores)
    else:
        print("No scores found in the specified folder.")

if __name__ == '__main__':
    main()