# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# 计算训练时间的函数
def cal_train_time(log_dicts, args):
    """分析训练时间并输出每个epoch的平均时间和标准差。

    Args:
        log_dicts (list): 包含每个epoch日志的字典列表。
        args (argparse.Namespace): 命令行参数。
    """
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        if not all_times:
            raise KeyError(
                'Please reduce the log interval in the config so that'
                'interval is less than iterations of one epoch.')
        epoch_ave_time = np.array(list(map(lambda x: np.mean(x), all_times)))
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print(f'slowest epoch {slowest_epoch + 1}, '
              f'average time is {epoch_ave_time[slowest_epoch]:.4f} s/iter')
        print(f'fastest epoch {fastest_epoch + 1}, '
              f'average time is {epoch_ave_time[fastest_epoch]:.4f} s/iter')
        print(f'time std over epochs is {std_over_epoch:.4f}')
        print(f'average iter time: {np.mean(epoch_ave_time):.4f} s/iter\n')


# 绘制曲线的函数
def plot_curve(log_dicts, args):
    """绘制训练过程中的指标曲线。

    Args:
        log_dicts (list): 包含每个epoch日志的字典列表。
        args (argparse.Namespace): 命令行参数。
    """
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # 如果没有提供图例，则使用 {filename}_{key} 作为图例
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    # TODO: 支持动态评估间隔（例如 RTMDet）在绘制 mAP 时
    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            if metric not in log_dict[epochs[int(args.eval_interval) - 1]]:
                if 'mAP' in metric:
                    raise KeyError(
                        f'{args.json_logs[i]} does not contain metric '
                        f'{metric}. Please check if "--no-validate" is '
                        'specified when you trained the model. Or check '
                        f'if the eval_interval {args.eval_interval} in args '
                        'is equal to the eval_interval during training.')
                raise KeyError(
                    f'{args.json_logs[i]} does not contain metric {metric}. '
                    'Please reduce the log interval in the config so that '
                    'interval is less than iterations of one epoch.')

            if 'mAP' in metric:
                xs = []
                ys = []
                for epoch in epochs:
                    ys += log_dict[epoch][metric]
                    if log_dict[epoch][metric]:
                        xs += [epoch]
                plt.xlabel('epoch')
                plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
            else:
                xs = []
                ys = []
                for epoch in epochs:
                    iters = log_dict[epoch]['step']
                    xs.append(np.array(iters))
                    ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                plt.xlabel('iter')
                plt.plot(
                    xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
            plt.legend()
        if args.title is not None:
            plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


# 添加绘图参数的解析器
def add_plot_parser(subparsers):
    """为绘图任务添加参数解析器。

    Args:
        subparsers (argparse._SubParsersAction): 子解析器对象。
    """
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['bbox_mAP'],
        help='the metric that you want to plot')
    parser_plt.add_argument(
        '--start-epoch',
        type=str,
        default='1',
        help='the epoch that you want to start')
    parser_plt.add_argument(
        '--eval-interval',
        type=str,
        default='1',
        help='the eval interval when training')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)


# 添加计算时间参数的解析器
def add_time_parser(subparsers):
    """为计算训练时间任务添加参数解析器。

    Args:
        subparsers (argparse._SubParsersAction): 子解析器对象。
    """
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='include the first value of every epoch when computing '
        'the average time')


# 解析命令行参数
def parse_args():
    """解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的命令行参数。
    """
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # 目前仅支持绘制曲线和计算平均训练时间
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args


# 加载JSON日志文件
def load_json_logs(json_logs):
    """加载并将json日志转换为日志字典，键为epoch，值为子字典。

    Args:
        json_logs (list): JSON日志文件路径列表。

    Returns:
        list: 包含每个epoch日志的字典列表。
    """
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            epoch = 1
            for i, line in enumerate(log_file):
                log = json.loads(line.strip())
                val_flag = False
                # 跳过仅包含一个键的行
                if not len(log) > 1:
                    continue

                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)

                for k, v in log.items():
                    if '/' in k:
                        log_dict[epoch][k.split('/')[-1]].append(v)
                        val_flag = True
                    elif val_flag:
                        continue
                    else:
                        log_dict[epoch][k].append(v)

                if 'epoch' in log.keys():
                    epoch = log['epoch']

    return log_dicts


# 主函数
def main():
    """主函数，解析参数并执行相应的任务。"""
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    log_dicts = load_json_logs(json_logs)

    eval(args.task)(log_dicts, args)


if __name__ == '__main__':
    main()



    # 2. 使用示例
    # 示例 1：绘制边界框平均精度
    # python analyze_logs.py plot_curve train_log_1.json train_log_2.json --keys bbox_mAP --title "Bounding Box mAP Curve" --legend "Log 1" "Log 2" --out bbox_mAP_curve.png

    # 示例 2：绘制 损失值
    # python analyze_logs.py plot_curve train_log_1.json train_log_2.json --keys loss --title "Loss Curve" --legend "Log 1" "Log 2" --out loss_curve.png

    # 示例 3：绘制 学习率
    # python analyze_logs.py plot_curve train_log_1.json train_log_2.json --keys lr --title "Learning Rate Curve" --legend "Log 1" "Log 2" --out lr_curve.png

    # 示例 4：绘制 训练时间
    # python analyze_logs.py cal_train_time train_log_1.json train_log_2.json --include-outliers

    #示例 4：绘制多个指标
    # python analyze_logs.py plot_curve train_log_1.json train_log_2.json --keys bbox_mAP loss lr --title "Multiple Metrics Curve" --legend "Log 1" "Log 2" --out multiple_metrics_curve.png