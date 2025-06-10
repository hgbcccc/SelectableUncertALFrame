import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.registry import MODELS
from mmdet.models.backbones.mobilenet_v2 import MobileNetV2

@MODELS.register_module()
class MultispectralMobileNetV2(MobileNetV2):
    """支持多光谱输入的MobileNetV2
    
    Args:
        in_channels (int): 输入通道数. 默认为4.
        其他参数与MobileNetV2相同.
    """

    def __init__(self,
                 in_channels=4,
                 widen_factor=1.,
                 out_indices=(1, 2, 4, 7),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):
        # 调用父类的__init__
        super().__init__(
            widen_factor=widen_factor,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            with_cp=with_cp,
            pretrained=pretrained,
            init_cfg=init_cfg)

        # 重新定义第一个卷积层以支持多通道输入
        self.conv1 = ConvModule(
            in_channels=in_channels,  # 修改输入通道数
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def init_weights(self):
        """初始化权重，特别处理预训练权重的加载"""
        if self.init_cfg is not None and self.init_cfg['type'] == 'Pretrained':
            # 加载预训练权重
            checkpoint = self._load_checkpoint()
            
            if checkpoint is not None:
                state_dict = checkpoint['state_dict']
                # 获取预训练模型的第一层权重
                pretrained_conv1_weight = state_dict['conv1.weight']
                
                # 初始化新的权重
                current_conv1_weight = self.conv1.conv.weight
                
                # 复制RGB通道的权重
                current_conv1_weight.data[:, :3] = pretrained_conv1_weight
                
                # 对于额外的通道，使用RGB通道的平均值进行初始化
                if current_conv1_weight.size(1) > 3:
                    avg_weight = pretrained_conv1_weight.mean(dim=1, keepdim=True)
                    current_conv1_weight.data[:, 3:] = avg_weight.repeat(
                        1, current_conv1_weight.size(1) - 3, 1, 1)
                
                # 删除conv1的权重，因为我们已经手动处理了
                if 'conv1.weight' in state_dict:
                    del state_dict['conv1.weight']
                if 'conv1.bias' in state_dict:
                    del state_dict['conv1.bias']
                
                # 加载其他层的权重
                self.load_state_dict(state_dict, strict=False)
        else:
            # 使用默认的初始化方法
            super().init_weights() 