import argparse
import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from sual.utils.utils import ( setup_work_dir,  # 设置工作目录
                            check_resume_state,  # 检查是否需要恢复训练
                            load_performance_history,  # 加载性能历史
                            update_checkpoint_state,  # 更新检查点状态
                            create_checkpoint_file,  # 创建检查点文件
                            save_round_stats,  # 保存本轮统计信息
                            update_dataset,  # 更新数据集
                            init_performance_history, # 初始化性能历史
                            update_performance_history, # 更新性能历史
                            process_unlabeled_data,  # 处理未标注数据
                            select_samples, # 选择样本
                            train_model) # 训练模型


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='主动学习训练')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--work-dir', help='工作目录')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='覆盖配置文件中的选项')
    parser.add_argument('--resume', action='store_true', help='是否从中断处恢复训练')
    args = parser.parse_args()
    return args


def run_active_learning_training(cfg: Config, args: argparse.Namespace):
    """执行主动学习训练"""
    # 设置工作目录
    work_dir = setup_work_dir(cfg, args)
    
    # 初始化日志
    logger = MMLogger.get_current_instance()
    logger.info(f"工作目录: {work_dir}")
    
    # 初始化性能历史
    performance_history = init_performance_history()
    
    # 检查是否需要恢复训练 
    max_iterations = cfg.active_learning.get('max_iterations', 10)
    start_round = 1
    if args.resume:
        start_round = check_resume_state(work_dir, max_iterations, logger)
        load_performance_history(work_dir, performance_history, logger)
    
    # 开始主动学习循环
    for round_num in range(start_round, max_iterations + 1):
        logger.info(f"\n开始第 {round_num}/{max_iterations} 轮主动学习...")
        
        # 创建轮次工作目录
        iter_work_dir = work_dir / f"round_{round_num}"
        iter_work_dir.mkdir(exist_ok=True)
        cfg.work_dir = str(iter_work_dir)
        
        # 创建检查点状态文件
        create_checkpoint_file(iter_work_dir)
        
        # 训练和评估
        logger.info(f"第 {round_num} 轮训练开始...")
        _, eval_results = train_model(cfg, work_dir, round_num, logger)
        update_checkpoint_state(iter_work_dir, 'training_done')
        
        # 处理未标注数据
        logger.info(f"第 {round_num} 轮推理开始...")
        train_stats, unlabeled_results = process_unlabeled_data(cfg, work_dir, round_num, logger)
        update_checkpoint_state(iter_work_dir, 'inference_done')
        
        # 选择样本
        logger.info(f"第 {round_num} 轮样本选择开始...")
        selected_samples = select_samples(train_stats, unlabeled_results, cfg, logger)
        update_checkpoint_state(iter_work_dir, 'selection_done')
        
        # 更新数据集和性能历史
        logger.info(f"第 {round_num} 轮更新数据集...")
        current_stats = update_dataset(selected_samples, cfg, logger)
        update_performance_history(performance_history, current_stats, eval_results, round_num)
        
        # 保存本轮统计信息
        save_round_stats(work_dir, round_num, selected_samples, eval_results, performance_history, logger, cfg)
        update_checkpoint_state(iter_work_dir, 'completed')
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        
        logger.info(f"第 {round_num} 轮已完成")

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # 保存配置文件
    work_dir = setup_work_dir(cfg, args)
    cfg.dump(work_dir / 'config.py')
    
    run_active_learning_training(cfg, args)

if __name__ == '__main__':
    import sys

    # sys.argv = ['sual/active_learnning_train_loop.py', 'custom_config/faster-rcnn_margin_default.py', 
    #             '--work-dir', 'work_dirs/faster-rcnn_margin_default']
    main()   
    # python sual/train.py custom_config/cascade-rcnn_ssc.py --work-dir work_dirs/cascade-rcnn_ssc_16_200 
    # python sual/train.py custom_config/cascade-rcnn_mus_cdb.py --work-dir work_dirs/cascade-rcnn_mus_16_200
    # python sual/train.py custom_config/faster-rcnn_mus_cdb.py --work-dir work_dirs/faster-rcnn_mus_16_200 
    # python sual/train.py custom_config_retinanet/retinanet_mus_cdb_16_200.py --work-dir work_dirs/retinanet_mus_16_200 
    # python sual/train.py fcos.py --work-dir work_dirs/fcos_ssc_16_200
    # python sual/train.py custom_config/cascade-rcnn_margin.py --work-dir new_work_dirs/cascade-rcnn_margin_16_200 --resume
    # python sual/train.py custom_config/cascade-rcnn_entropy.py --work-dir new_work_dirs/cascade-rcnn_entropy_16_200 --resume
    # python sual/train.py custom_config/cascade-rcnn_ssc.py --work-dir new_work_dirs/cascade-rcnn_ssc_16_200 --resume
    # python sual/train.py custom_config/cascade-rcnn_sor.py --work-dir new_work_dirs/cascade-rcnn_sor_16_200 --resume