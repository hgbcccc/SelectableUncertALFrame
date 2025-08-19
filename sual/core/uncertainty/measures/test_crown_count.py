import numpy as np
import matplotlib.pyplot as plt

def calculate_crown_count_score(crown_count, n_batch, sigma_batch):
    """计算树冠控制系数
    
    Args:
        crown_count: 当前图片的检测框数量
        n_batch: 批次平均检测框数量
        sigma_batch: 批次检测框数量标准差
        
    Returns:
        float: 树冠控制系数
    """
    # 放宽边界阈值
    upper_threshold = n_batch * 2.0
    lower_threshold = n_batch * 0.3
    
    # 计算基础高斯得分
    normalized_diff = (crown_count - n_batch) / (sigma_batch + 1e-6)
    base_score = np.exp(-0.5 * normalized_diff ** 2)
    
    # 含过渡和平滑优化
    transition_width = 25
    slope = 8.0
    
    # 上界平滑过渡
    upper_smooth = 1 / (1 + np.exp((crown_count - upper_threshold + transition_width/2)/transition_width*slope))
    
    # 下界平滑过渡  
    lower_smooth = 1 / (1 + np.exp((-crown_count + lower_threshold + transition_width/2)/transition_width*slope))
    
    # 综合得分
    crown_count_score = base_score * upper_smooth * lower_smooth
    
    # 分数截断
    crown_count_score = max(0.00, crown_count_score)
    
    return crown_count_score

def simulate_crown_count_scores():
    """模拟不同检测框数量下的树冠控制系数"""
    # 设置参数
    n_batch = 300.0
    sigma_batch = 0.0  # 标准差为0的情况
    
    # 模拟5张图片，检测框数量分别为100, 200, 300, 400, 500
    crown_counts_varied = [100, 200, 300, 400, 500]
    scores_varied = [calculate_crown_count_score(count, n_batch, sigma_batch) for count in crown_counts_varied]
    
    # 模拟5张图片，检测框数量都是300
    crown_counts_same = [300, 300, 300, 300, 300]
    scores_same = [calculate_crown_count_score(count, n_batch, sigma_batch) for count in crown_counts_same]
    
    # 打印结果
    print("检测框数量不同的情况 (sigma_batch = 0.0):")
    for count, score in zip(crown_counts_varied, scores_varied):
        print(f"检测框数量: {count}, 树冠控制系数: {score}")
    
    print("\n检测框数量相同的情况 (sigma_batch = 0.0):")
    for count, score in zip(crown_counts_same, scores_same):
        print(f"检测框数量: {count}, 树冠控制系数: {score}")
    
    # 使用小的非零sigma_batch值
    sigma_batch = 10.0
    scores_varied_small_sigma = [calculate_crown_count_score(count, n_batch, sigma_batch) for count in crown_counts_varied]
    
    print("\n检测框数量不同的情况 (sigma_batch = 10.0):")
    for count, score in zip(crown_counts_varied, scores_varied_small_sigma):
        print(f"检测框数量: {count}, 树冠控制系数: {score}")
    
    # 详细计算过程
    print("\n详细计算过程 (sigma_batch = 10.0):")
    for count in crown_counts_varied:
        # 计算基础高斯得分
        normalized_diff = (count - n_batch) / (sigma_batch + 1e-6)
        base_score = np.exp(-0.5 * normalized_diff ** 2)
        
        # 放宽边界阈值
        upper_threshold = n_batch * 2.0
        lower_threshold = n_batch * 0.3
        
        # 含过渡和平滑优化
        transition_width = 25
        slope = 8.0
        
        # 上界平滑过渡
        upper_smooth = 1 / (1 + np.exp((count - upper_threshold + transition_width/2)/transition_width*slope))
        
        # 下界平滑过渡  
        lower_smooth = 1 / (1 + np.exp((-count + lower_threshold + transition_width/2)/transition_width*slope))
        
        # 综合得分
        crown_count_score = base_score * upper_smooth * lower_smooth
        
        print(f"检测框数量: {count}")
        print(f"  normalized_diff: {normalized_diff}")
        print(f"  base_score: {base_score}")
        print(f"  upper_smooth: {upper_smooth}")
        print(f"  lower_smooth: {lower_smooth}")
        print(f"  crown_count_score: {crown_count_score}")
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(crown_counts_varied, scores_varied, color='skyblue')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('检测框数量不同的情况 (sigma_batch = 0.0)')
    plt.xlabel('检测框数量')
    plt.ylabel('树冠控制系数')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(crown_counts_same, scores_same, color='lightgreen')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('检测框数量相同的情况 (sigma_batch = 0.0)')
    plt.xlabel('检测框数量')
    plt.ylabel('树冠控制系数')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('crown_count_scores.png')
    plt.show()
    
    # 测试不同的sigma_batch值
    sigma_values = [0.0, 10.0, 50.0, 100.0, 200.0]
    plt.figure(figsize=(12, 8))
    
    for sigma in sigma_values:
        scores = [calculate_crown_count_score(count, n_batch, sigma) for count in crown_counts_varied]
        plt.plot(crown_counts_varied, scores, marker='o', label=f'sigma={sigma}')
    
    plt.axvline(x=n_batch, color='r', linestyle='--', alpha=0.5)
    plt.title('不同sigma值下的树冠控制系数')
    plt.xlabel('检测框数量')
    plt.ylabel('树冠控制系数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('crown_count_scores_sigma.png')
    plt.show()

def test_parameter_sensitivity():
    """测试不同参数对树冠控制系数平滑度的影响"""
    # 基础参数
    n_batch = 300.0
    
    # 创建检测框数量范围
    crown_counts = np.linspace(0, 600, 601)  # 从0到600，共601个点
    
    # 测试不同的sigma_batch值
    sigma_values = [0.0, 10.0, 50.0, 100.0, 200.0, 500.0]
    
    plt.figure(figsize=(15, 10))
    
    for sigma in sigma_values:
        scores = [calculate_crown_count_score(count, n_batch, sigma) for count in crown_counts]
        plt.plot(crown_counts, scores, label=f'sigma={sigma}')
    
    plt.axvline(x=n_batch, color='r', linestyle='--', alpha=0.5, label='平均值')
    plt.title('不同sigma值下的树冠控制系数')
    plt.xlabel('检测框数量')
    plt.ylabel('树冠控制系数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('crown_count_scores_sigma_range.png')
    plt.show()
    
    # 测试不同的transition_width值
    transition_widths = [5, 10, 25, 50, 100]
    sigma_batch = 100.0  # 使用一个适中的sigma值
    
    plt.figure(figsize=(15, 10))
    
    for width in transition_widths:
        # 临时修改transition_width
        original_transition_width = transition_width
        transition_width = width
        
        scores = [calculate_crown_count_score(count, n_batch, sigma_batch) for count in crown_counts]
        plt.plot(crown_counts, scores, label=f'transition_width={width}')
        
        # 恢复原始值
        transition_width = original_transition_width
    
    plt.axvline(x=n_batch, color='r', linestyle='--', alpha=0.5, label='平均值')
    plt.title('不同transition_width值下的树冠控制系数')
    plt.xlabel('检测框数量')
    plt.ylabel('树冠控制系数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('crown_count_scores_transition_width.png')
    plt.show()
    
    # 测试不同的slope值
    slopes = [1.0, 2.0, 4.0, 8.0, 16.0]
    sigma_batch = 100.0  # 使用一个适中的sigma值
    
    plt.figure(figsize=(15, 10))
    
    for s in slopes:
        # 临时修改slope
        original_slope = slope
        slope = s
        
        scores = [calculate_crown_count_score(count, n_batch, sigma_batch) for count in crown_counts]
        plt.plot(crown_counts, scores, label=f'slope={s}')
        
        # 恢复原始值
        slope = original_slope
    
    plt.axvline(x=n_batch, color='r', linestyle='--', alpha=0.5, label='平均值')
    plt.title('不同slope值下的树冠控制系数')
    plt.xlabel('检测框数量')
    plt.ylabel('树冠控制系数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('crown_count_scores_slope.png')
    plt.show()
    
    # 测试不同的上下界阈值
    threshold_multipliers = [(0.3, 2.0), (0.5, 1.5), (0.7, 1.3), (0.9, 1.1)]
    sigma_batch = 100.0  # 使用一个适中的sigma值
    
    plt.figure(figsize=(15, 10))
    
    for lower_mult, upper_mult in threshold_multipliers:
        # 临时修改阈值计算
        original_calculate_crown_count_score = calculate_crown_count_score
        
        def temp_calculate_crown_count_score(crown_count, n_batch, sigma_batch):
            # 计算基础高斯得分
            normalized_diff = (crown_count - n_batch) / (sigma_batch + 1e-6)
            base_score = np.exp(-0.5 * normalized_diff ** 2)
            
            # 放宽边界阈值
            upper_threshold = n_batch * upper_mult
            lower_threshold = n_batch * lower_mult
            
            # 含过渡和平滑优化
            transition_width = 25
            slope = 8.0
            
            # 上界平滑过渡
            upper_smooth = 1 / (1 + np.exp((crown_count - upper_threshold + transition_width/2)/transition_width*slope))
            
            # 下界平滑过渡  
            lower_smooth = 1 / (1 + np.exp((-crown_count + lower_threshold + transition_width/2)/transition_width*slope))
            
            # 综合得分
            crown_count_score = base_score * upper_smooth * lower_smooth
            
            # 确保得分不为负
            crown_count_score = max(0.00, crown_count_score)
            
            return crown_count_score
        
        calculate_crown_count_score = temp_calculate_crown_count_score
        
        scores = [calculate_crown_count_score(count, n_batch, sigma_batch) for count in crown_counts]
        plt.plot(crown_counts, scores, label=f'阈值: ({lower_mult}, {upper_mult})')
        
        # 恢复原始函数
        calculate_crown_count_score = original_calculate_crown_count_score
    
    plt.axvline(x=n_batch, color='r', linestyle='--', alpha=0.5, label='平均值')
    plt.title('不同上下界阈值下的树冠控制系数')
    plt.xlabel('检测框数量')
    plt.ylabel('树冠控制系数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('crown_count_scores_thresholds.png')
    plt.show()

if __name__ == "__main__":
    # 模拟不同检测框数量下的树冠控制系数
    simulate_crown_count_scores()
    
    # 测试参数敏感性
    test_parameter_sensitivity() 