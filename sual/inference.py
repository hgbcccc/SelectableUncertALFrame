import os
import argparse
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
import torch
import json
from tqdm import tqdm
from pathlib import Path
'''进行推理'''
def parse_args():
    parser = argparse.ArgumentParser(description='批量推理不同轮次的模型')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--base-dir', help='包含所有轮次模型权重的基础目录')
    parser.add_argument(
        '--img-dir',
        help='需要推理的图片目录')
    parser.add_argument(
        '--out-dir',
        help='输出结果的目录')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='推理使用的设备')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='置信度阈值')
    parser.add_argument(
        '--rounds',
        type=int,
        default=16,
        help='要分析的轮次数量')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载配置文件
    cfg = Config.fromfile(args.config)
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 获取所有图片文件
    img_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        img_files.extend(list(Path(args.img_dir).glob(f'*{ext}')))
    img_files = sorted(img_files)
    
    print(f"找到 {len(img_files)} 张图片")
    
    # 遍历每一轮的模型
    for round_idx in range(1, args.rounds + 1):
        print(f"\n处理第 {round_idx} 轮...")
        
        # 构建模型权重文件路径
        weight_path = os.path.join(args.base_dir, f'round_{round_idx}', 'epoch_3.pth')
        if not os.path.exists(weight_path):
            print(f"未找到模型权重文件: {weight_path}")
            continue
        
        # 创建当前轮次的输出目录
        round_out_dir = os.path.join(args.out_dir, f'round_{round_idx}')
        os.makedirs(round_out_dir, exist_ok=True)
        
        # 初始化模型
        model = init_detector(cfg, weight_path, device=args.device)
        
        # 存储所有图片的结果
        round_results = {}
        
        # 处理每张图片
        for img_file in tqdm(img_files, desc=f"Round {round_idx} Processing"):
            # 进行推理
            result = inference_detector(model, str(img_file))
            
            # 获取预测结果
            pred_instances = result.pred_instances
            
            # 将结果转换为numpy数组并过滤低置信度预测
            if pred_instances.scores.shape[0] > 0:
                valid_mask = pred_instances.scores >= args.score_thr
                bboxes = pred_instances.bboxes[valid_mask].cpu().numpy()
                scores = pred_instances.scores[valid_mask].cpu().numpy()
                labels = pred_instances.labels[valid_mask].cpu().numpy()
            else:
                bboxes = []
                scores = []
                labels = []
            
            # 保存结果
            img_result = {
                'filename': img_file.name,
                'predictions': {
                    'boxes': bboxes.tolist(),
                    'scores': scores.tolist(),
                    'labels': labels.tolist()
                }
            }
            
            # 将结果保存到字典
            round_results[img_file.name] = img_result
        
        # 保存当前轮次的所有结果
        results_file = os.path.join(round_out_dir, 'inference_results.json')
        with open(results_file, 'w') as f:
            json.dump(round_results, f, indent=2)
        
        print(f"第 {round_idx} 轮结果已保存到: {results_file}")
        
        # 释放GPU内存
        del model
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()