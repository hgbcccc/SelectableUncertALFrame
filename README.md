## 🎯SelectableUncertALFrame：集成多种采样策略的目标检测主动学习框架


## 🌟简介

SelectableUncertALFrame 是基于 MMDetection 的多采样策略目标检测主动学习框架。现有的目标检测主动学习框架往往仅着眼于单一采样策略的训练和验证，这致使不同策略间的公平比较颇具难度。本框架成功地整合了多种主流采样策略，为研究人员给予了全面的目标检测主动学习策略评估和比较实验的条件。

## 💪核心优势

- **集成多种采样策略**：将多种目标检测主动学习采样策略统一到同一框架下，[详情点击](docs\主动学习查询策略.md)
- **深度集成 OpenMMLab**：基于 MMDetection 开发，继承了其高效性和扩展性
- **统一接口设计**：提供统一的策略接口，便于研究人员进行策略对比和算法改进
- **高度可复用**：支持快速迁移和二次开发

## 🔧环境要求

- Windows system
- Python 3.9.20
- Torch 2.0.1+cu118
- Torchvision 0.15.2
- Torchaudio 1.8.0
- MMEngine 0.10.5
- MMDetection 3.1.0

## 📥安装

```
# 创建并激活 conda 环境
conda create -n sual python=3.9.20 -y
conda activate sual
pip install https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp39-cp39-win_amd64.whl
pip install torchvision 0.15.2 torchaudio 1.8.0 

# 克隆特定版本的 MMDetection
git clone -b v3.1.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -e .

#将sual放于mmdetection目录下,作为主动学习扩展的插件
git clone https://github.com/your-repo/sual.git
cd sual 
pip install -e .
```

## 🚀 快速开始

```
## 准备主动学习数据集 10%用于初始训练students模型，90%作为未标注数据
## 配置文件参考custom_config

!python tools/prepare_active_dataset.py \
    /path/to/data_root \  # 该数据集类型必须为MS COCO形式
    /path/to/save_dir \
    --train-ratio 0.04 \
    --val-ratio 0.01 \
    --seed 42
## 得到如下数据集目录

# save_dir/
# ├── images_labeled_train/
# ├── images_labeled_val/
# ├── images_unlabeled/
# └── annotations/
#     ├── instances_labeled_train.json
#     ├── instances_labeled_val.json
#     └── instances_unlabeled.json



## 开始主动学习训练

!python tools/al_train.py /path/to/config.py --work-dir  /path/to/savedir 
```

## 📚 文档

* **[详细安装指南]()**
* **[基础教程]()**
* **[进阶指南]()**
* **[API 文档]()**

## 🤝 贡献指南

我们欢迎任何形式的贡献：

* 🐛 提交问题和建议
* 📝 改进文档
* 🔧 提交代码修复
* ✨ 提出新功能

## 📄 许可证

本项目采用 **Apache 2.0 许可证**

