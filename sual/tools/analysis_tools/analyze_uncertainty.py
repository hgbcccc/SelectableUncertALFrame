# tools/analyze_uncertainty.py
import os
import argparse
from mmdet.apis import init_detector
from sual.apis.inference import analyze_image_uncertainty
from sual.utils.vis_utils import plot_uncertainty_distribution
from sual.utils.metrics_utils import calculate_statistics

def main():
    parser = argparse.ArgumentParser(description='分析图片检测的不确定性')
    parser.add_argument('img', help='图片路径')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='模型文件路径')
    parser.add_argument('--out-dir', help='输出目录', default='outputs')
    parser.add_argument('--metrics', nargs='+', 
                       default=['confidence', 'entropy'],
                       help='使用的不确定性度量方法')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 初始化模型
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    
    # 分析不确定性
    uncertainty_results = analyze_image_uncertainty(
        model, 
        args.img,
        metrics=args.metrics
    )
    
    # 输出结果
    print("\n=== 不确定性分析结果 ===")
    for metric_name, value in uncertainty_results.items():
        print(f"{metric_name}: {value:.3f}")
    
    # 保存可视化结果
    plot_uncertainty_distribution(
        list(uncertainty_results.values()),
        save_path=os.path.join(args.out_dir, 'uncertainty_distribution.png')
    )

if __name__ == '__main__':
    main()