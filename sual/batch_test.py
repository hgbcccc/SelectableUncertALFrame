# batch_test.py

import argparse
import os
from pathlib import Path
import pandas as pd
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo
import datetime
import json
import time
'''脚本的作用是批量测试模型，并保存测试结果'''
def parse_args():
    parser = argparse.ArgumentParser(description='批量测试模型')
    parser.add_argument('--test-work-dir', default='work_dirs/test_results', 
                       help='测试结果保存目录')
    return parser.parse_args()

def find_checkpoints(work_dir):
    """查找工作目录下所有轮次的检查点"""
    checkpoints = []
    work_dir = Path(work_dir)
    
    # 遍历round_x目录
    for round_dir in work_dir.glob('round_*'):
        if not round_dir.is_dir():
            continue
            
        # 检查epoch_1.pth是否存在
        ckpt_path = round_dir / 'epoch_1.pth'
        if ckpt_path.exists():
            round_num = int(round_dir.name.split('_')[1])
            checkpoints.append((round_num, str(ckpt_path)))
            
    return sorted(checkpoints, key=lambda x: x[0])

def test_model(config_file, checkpoint, work_dir):
    """测试单个模型"""
    cfg = Config.fromfile(config_file)
    
    # 设置工作目录和检查点
    cfg.work_dir = work_dir
    cfg.load_from = checkpoint
    
    # 创建runner并测试
    runner = Runner.from_cfg(cfg)
    results = runner.test()
    return results

def main():
    args = parse_args()
    setup_cache_size_limit_of_dynamo()
    
    # 读取配置列表
    configs = [
        {
            'config': 'custom_config/faster-rcnn_ssc_combinatorial.py',
            'base_dir': 'work_dirs/al_faster-rcnn_ssc_combinatorial',
        },
        {
            'config': 'custom_config/faster-rcnn_ssc_Default.py',
            'base_dir': 'work_dirs/al_faster-rcnn_ssc_default',
        },
        {
            'config': 'custom_config/faster-rcnn_ssc_RL_wasserstein.py',
            'base_dir': 'work_dirs/al_faster-rcnn_ssc_rl_wasserstein',
        },
        {
            'config': 'custom_config/faster-rcnn_ssc_RL_kl.py',
            'base_dir': 'work_dirs/al_faster-rcnn_ssc_rl_kl',
        },
        {
            'config': 'custom_config/faster-rcnn_ssc_Wasserstein.py',
            'base_dir': 'work_dirs/al_faster-rcnn_ssc_wasserstein',
        },
        {
            'config': 'custom_config/faster-rcnn_basic_default.py',
            'base_dir': 'work_dirs/faster-rcnn_basic_default',
        },
        {
            'config': 'custom_config/faster-rcnn_entropy.py',
            'base_dir': 'work_dirs/faster-rcnn_entropy',
        },
        {
            'config': 'custom_config/faster-rcnn_least_confid.py',
            'base_dir': 'work_dirs/faster-rcnn_least_confid',
        },
        {
            'config': 'custom_config/faster-rcnn_sor.py',
            'base_dir': 'work_dirs/faster-rcnn_sor',
        },
        {
            'config': 'custom_config/faster-rcnn_margin_default.py',
            'base_dir': 'work_dirs/faster-rcnn_margin_default',
        },
    ]
    
    # 创建结果保存目录
    os.makedirs(args.test_work_dir, exist_ok=True)
    
    # 准备DataFrame数据
    results_data = []
    
    # 遍历每个配置
    for config in configs:
        config_file = config['config']
        base_dir = config['base_dir']
        
        print(f"\n开始测试 {base_dir}")
        
        # 查找所有检查点
        checkpoints = find_checkpoints(base_dir)
        if not checkpoints:
            print(f"在 {base_dir} 中未找到检查点")
            continue
            
        # 测试每个检查点
        for round_num, checkpoint in checkpoints:
            print(f"测试轮次 {round_num}")
            
            try:
                # 设置测试工作目录
                test_dir = os.path.join(args.test_work_dir, 
                                      f"{Path(base_dir).name}_round_{round_num}")
                
                # 运行测试
                results = test_model(config_file, checkpoint, test_dir)
                
                # 记录结果
                result_dict = {
                    'model_name': Path(base_dir).name,
                    'round': round_num,
                    'config_file': config_file,
                    'checkpoint': checkpoint,
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'bbox_mAP': results.get('bbox_mAP', 0.0),
                    'bbox_mAP_50': results.get('bbox_mAP_50', 0.0),
                    'bbox_mAP_75': results.get('bbox_mAP_75', 0.0),
                    'bbox_mAP_s': results.get('bbox_mAP_s', 0.0),
                    'bbox_mAP_m': results.get('bbox_mAP_m', 0.0),
                    'bbox_mAP_l': results.get('bbox_mAP_l', 0.0)
                }
                
                results_data.append(result_dict)
                
                # 保存详细结果
                detail_file = os.path.join(test_dir, 'test_results.json')
                with open(detail_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
            except Exception as e:
                print(f"测试失败: {e}")
                continue
    
    # 保存CSV结果
    if results_data:
        df = pd.DataFrame(results_data)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(args.test_work_dir, f'test_results_{timestamp}.csv')
        df.to_csv(csv_file, index=False)
        print(f"\n结果已保存到: {csv_file}")
        
        # 打印汇总信息
        print("\n性能汇总:")
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            print(f"\n{model}:")
            print(f"- 最终mAP: {model_df['bbox_mAP'].iloc[-1]:.4f}")
            print(f"- 最佳mAP: {model_df['bbox_mAP'].max():.4f}")
            print(f"- 最佳mAP_50: {model_df['bbox_mAP_50'].max():.4f}")
            print(f"- 最佳mAP_75: {model_df['bbox_mAP_75'].max():.4f}")
    else:
        print("没有成功的测试结果")

if __name__ == '__main__':
    main()