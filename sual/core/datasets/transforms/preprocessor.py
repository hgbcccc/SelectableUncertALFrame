# sual/core/datasets/transforms/preprocessor.py

from typing import Optional, Sequence, Union, Dict
import torch
import torch.nn.functional as F
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS
from mmengine.structures import PixelData
import torch.nn as nn

@MODELS.register_module()
class MultiSpectralDetDataPreprocessor(nn.Module):
    """多光谱图像检测数据预处理器"""
    
    def __init__(self,
                 mean: Optional[Sequence[float]] = None,
                 std: Optional[Sequence[float]] = None,
                 num_channels: int = 4,  # 默认4通道
                 bgr_to_rgb: bool = True,
                 rgb_to_bgr: bool = False,
                 pad_size_divisor: int = 32,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 boxtype2tensor: bool = True,
                 device: str = 'cuda:0'):
        # 首先调用nn.Module的初始化
        super().__init__()
        

        self.num_channels = num_channels
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.bgr_to_rgb = bgr_to_rgb
        self.rgb_to_bgr = rgb_to_bgr
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        self.pad_seg = pad_seg
        self.seg_pad_value = seg_pad_value
        self.boxtype2tensor = boxtype2tensor
        self.device = device


        # 验证并注册mean和std
        if mean is not None:
            if not isinstance(mean, Sequence):
                mean = [mean] * num_channels
            assert len(mean) == num_channels, \
                f'mean必须有{num_channels}个值以匹配通道数，但得到{len(mean)}个值'
            self.register_buffer('mean',
                               torch.tensor(mean).view(-1, 1, 1))
        else:
            self.register_buffer('mean', None)
        
        if std is not None:
            if not isinstance(std, Sequence):
                std = [std] * num_channels
            assert len(std) == num_channels, \
                f'std必须有{num_channels}个值以匹配通道数，但得到{len(std)}个值'
            self.register_buffer('std',
                               torch.tensor(std).view(-1, 1, 1))
        else:
            self.register_buffer('std', None)

        # # 验证并注册mean和std
        # if mean is not None:
        #     if not isinstance(mean, Sequence):
        #         mean = [mean] * num_channels
        #     assert len(mean) == num_channels, \
        #         f'mean必须有{num_channels}个值以匹配通道数，但得到{len(mean)}个值'
        #     self.register_buffer('mean',
        #                        torch.tensor(mean).view(-1, 1, 1))
        
        # if std is not None:
        #     if not isinstance(std, Sequence):
        #         std = [std] * num_channels
        #     assert len(std) == num_channels, \
        #         f'std必须有{num_channels}个值以匹配通道数，但得到{len(std)}个值'
        #     self.register_buffer('std',
        #                        torch.tensor(std).view(-1, 1, 1))

    def forward(self, data: dict, training: bool = False) -> dict:
        """处理多光谱数据"""
        batch_pad_shape = self._get_pad_shape(data)
        
        # 获取输入数据
        inputs = data['inputs']
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.stack(inputs)
        
        # 转换设备
        inputs = inputs.to(self.device)
        
        # 标准化
        if hasattr(self, 'mean') and hasattr(self, 'std'):
            inputs = (inputs - self.mean) / self.std
            
        # 填充
        if self.pad_size_divisor > 0:
            inputs = self._pad_to_size_divisor(inputs)
            
        # 处理数据样本
        data_samples = data.get('data_samples', None)
        if data_samples is not None:
            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })

            if self.boxtype2tensor:
                from mmdet.models.utils.misc import samplelist_boxtype2tensor
                samplelist_boxtype2tensor(data_samples)

            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

            if self.pad_seg and training:
                self.pad_gt_sem_seg(data_samples)

        return {'inputs': inputs, 'data_samples': data_samples}

    def _pad_to_size_divisor(self, inputs: torch.Tensor) -> torch.Tensor:
        """填充到指定大小"""
        pad_h = (self.pad_size_divisor -
                inputs.shape[-2] % self.pad_size_divisor) % self.pad_size_divisor
        pad_w = (self.pad_size_divisor -
                inputs.shape[-1] % self.pad_size_divisor) % self.pad_size_divisor
        if pad_h > 0 or pad_w > 0:
            inputs = F.pad(
                inputs, (0, pad_w, 0, pad_h),
                mode='constant',
                value=self.pad_value)
        return inputs

    def _get_pad_shape(self, data: dict) -> list:
        """获取填充形状"""
        inputs = data['inputs']
        if not isinstance(inputs, torch.Tensor):
            inputs = [input_.shape[-2:] for input_ in inputs]
        else:
            inputs = [inputs.shape[-2:]]
        return inputs