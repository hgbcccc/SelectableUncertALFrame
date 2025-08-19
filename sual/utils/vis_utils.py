# sual/utils/vis_utils.py
import matplotlib.pyplot as plt

def plot_uncertainty_distribution(uncertainties, save_path=None):
    """绘制不确定性分布图"""
    plt.figure(figsize=(10, 6))
    plt.hist(uncertainties, bins=50, edgecolor='black')
    plt.title('不确定性分布')
    plt.xlabel('不确定性值')
    plt.ylabel('数量')
    if save_path:
        plt.savefig(save_path)
    plt.close()

# sual/utils/metrics_utils.py
import numpy as np

def calculate_statistics(values):
    """计算基本统计量"""
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values)
    }