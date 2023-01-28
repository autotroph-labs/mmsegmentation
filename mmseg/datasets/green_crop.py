# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class GreenCropDataset(CustomDataset):
    
    CLASSES = ('background', 'green_crop')

    PALETTE = [[120, 120, 120], [6, 230, 230]]

    def __init__(self, **kwargs):
        super(GreenCropDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', reduce_zero_label=False, **kwargs)
