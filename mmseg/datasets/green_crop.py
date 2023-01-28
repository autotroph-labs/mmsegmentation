# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class GreenCropDataset(CustomDataset):
    
    CLASSES = ('green_crop',)

    PALETTE = [[0, 192, 64],]

    def __init__(self, **kwargs):
        super(GreenCropDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
