import os
from typing import Any, Dict

import torch.utils.data as data
import utils.utils_image as util


class DatasetIR(data.Dataset):
    def __init__(self, opt_dataset: Dict[str, Any]):
        super().__init__()

        self.opt = opt_dataset

        if self.opt['phase'] == 'train':
            self.patch_size = self.opt['H_size']
        self.n_channels = opt_dataset['n_channels']
#        self.sigma = opt_dataset['sigma']
        self.sigma = ['estimate']   # 对数据没有影响   2022.8.5

        self.name: str = os.path.basename(opt_dataset['dataroot_H'])
        self.img_paths = util.get_img_paths(opt_dataset['dataroot_H'])   # 参考图像数据
        self.img_paths2 = util.get_img_paths(opt_dataset['dataroot_L'])  # 低剂量CT图像数据

        self.count = 0
        self.tag: str = ""

    def __len__(self):
        return len(self.img_paths)
