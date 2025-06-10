# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, List

import mmengine.fileio as fileio
import numpy as np
import rasterio
import mmcv
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.builder import TRANSFORMS



@TRANSFORMS.register_module()
class LoadMultiSpectralTiffFromFile(BaseTransform):
    """Load a multispectral tiff image from file.

    Required Keys:
        - img_path

    Modified Keys:
        - img
        - img_shape
        - ori_shape
        - img_channel
        - img_metadata (optional)

    Args:
        to_float32 (bool): Whether to convert the loaded image to float32.
            Defaults to True.
        selected_bands (List[int], optional): Selected band indices.
            Defaults to None, which means loading all bands.
        normalize (bool): Whether to normalize each band. Defaults to True.
        ignore_empty (bool): Whether to allow loading empty image.
            Defaults to False.
        backend_args (dict, optional): Arguments to instantiate file backend.
            Defaults to None.
    """

    def __init__(self,
                 to_float32: bool = True,
                 selected_bands: Optional[List[int]] = None,
                 normalize: bool = True,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        super().__init__()
        self.to_float32 = to_float32
        self.selected_bands = selected_bands
        self.normalize = normalize
        self.ignore_empty = ignore_empty
        self.backend_args = backend_args

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load multispectral tiff image.

        Args:
            results (dict): Result dict from dataset.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename = results['img_path']
        try:
            with rasterio.open(filename) as src:
                # 读取指定波段或所有波段
                if self.selected_bands is not None:
                    img = src.read(self.selected_bands)
                else:
                    img = src.read()
                
                # 转换数据格式 (C,H,W) -> (H,W,C)
                img = np.transpose(img, (1, 2, 0))
                
                # 数据归一化
                if self.normalize:
                    img = img.astype(np.float32)
                    for i in range(img.shape[-1]):
                        band = img[..., i]
                        min_val = np.percentile(band, 1)
                        max_val = np.percentile(band, 99)
                        img[..., i] = np.clip(
                            (band - min_val) / (max_val - min_val), 
                            0, 1
                        )
                
                if self.to_float32 and not self.normalize:
                    img = img.astype(np.float32)

                # 保存图像元数据
                results['img_metadata'] = {
                    'transform': src.transform,
                    'crs': src.crs,
                    'bands': self.selected_bands or list(range(1, src.count + 1))
                }

        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise ValueError(f'Failed to load {filename}: {str(e)}')

        if img is None:
            if self.ignore_empty:
                return None
            else:
                raise ValueError(f'Failed to load {filename}')

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['img_channel'] = img.shape[2]
        
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                   f'to_float32={self.to_float32}, '
                   f'selected_bands={self.selected_bands}, '
                   f'normalize={self.normalize}, '
                   f'ignore_empty={self.ignore_empty}, '
                   f'backend_args={self.backend_args})')
        return repr_str


