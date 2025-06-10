# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class TreeAIDataset(BaseDetDataset):
    """Dataset for TreeAI."""

    METAINFO = {
        'classes': (
            'betula papyrifera', 'tsuga canadensis', 'picea abies', 'acer saccharum', 'betula sp.', 
            'pinus sylvestris', 'picea rubens', 'betula alleghaniensis', 'larix decidua', 'fagus grandifolia', 
            'picea sp.', 'fagus sylvatica', 'dead tree', 'acer pensylvanicum', 'populus balsamifera', 
            'quercus ilex', 'quercus robur', 'pinus strobus', 'larix laricina', 'larix gmelinii', 
            'pinus pinea', 'populus grandidentata', 'pinus montezumae', 'abies alba', 'betula pendula', 
            'pseudotsuga menziesii', 'fraxinus nigra', 'dacrydium cupressinum', 'cedrus libani', 'acer pseudoplatanus', 
            'pinus elliottii', 'cryptomeria japonica', 'pinus koraiensis', 'abies holophylla', 'alnus glutinosa', 
            'fraxinus excelsior', 'coniferous', 'eucalyptus globulus', 'pinus nigra', 'quercus rubra', 
            'tilia europaea', 'abies firma', 'acer sp.', 'metrosideros umbellata', 'acer rubrum', 
            'picea mariana', 'abies balsamea', 'castanea sativa', 'tilia cordata', 'populus sp.', 
            'crataegus monogyna', 'quercus petraea', 'acer platanoides', 
        ),
        'palette': [
            (196, 149, 217), (86, 253, 65), (65, 171, 173), (52, 162, 43), (76, 45, 166), 
            (203, 96, 30), (26, 254, 37), (3, 54, 225), (152, 151, 244), (205, 25, 167), 
            (42, 154, 150), (122, 123, 247), (224, 10, 1), (224, 158, 143), (134, 225, 227), 
            (19, 13, 134), (6, 129, 150), (15, 188, 84), (127, 116, 139), (217, 220, 29), 
            (161, 151, 57), (122, 152, 219), (131, 29, 59), (98, 59, 38), (48, 239, 124), 
            (19, 131, 19), (197, 165, 154), (180, 157, 230), (232, 198, 66), (73, 185, 22), 
            (33, 243, 103), (248, 81, 208), (133, 140, 253), (37, 178, 109), (50, 150, 166), 
            (44, 175, 76), (245, 171, 92), (85, 187, 84), (67, 101, 132), (165, 76, 237), 
            (102, 171, 209), (69, 200, 147), (39, 155, 151), (19, 106, 114), (218, 167, 243), 
            (106, 242, 237), (235, 34, 172), (50, 86, 231), (240, 234, 19), (171, 10, 67), 
            (191, 71, 155), (251, 231, 83), (132, 75, 112), 
        ]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
