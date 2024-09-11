import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples


# @register('sr-implicit-paired')
# class SRImplicitPaired(Dataset):
#
#     def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
#         self.dataset = dataset
#         self.inp_size = inp_size
#         self.augment = augment
#         self.sample_q = sample_q
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         img_lr, img_lr_gt, mask, img_hr_gt = self.dataset[idx]
#         s = img_hr_gt.shape[-2] #// img_lr.shape[-2] # assume int scale
#
#
#         h_lr, w_lr = img_lr.shape[-2:]
#         img_hr = img_hr_gt[:, :h_lr * s, :w_lr * s]
#         crop_lr, crop_hr = img_lr_gt, img_hr
#
#
#         lr_coord, lr_rgb = to_pixel_samples(crop_lr.contiguous())
#         hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
#         hr_coord_full = hr_coord
#
#         if self.sample_q is not None:
#             sample_lst = np.random.choice(
#                 len(hr_coord_full), self.sample_q, replace=False)
#             hr_coord = hr_coord_full[sample_lst]
#             hr_rgb = hr_rgb[sample_lst]
#             # mask = mask[sample_lst]
#
#
#         return {
#             'inp': img_lr,
#             'lr_gt': img_lr_gt,
#             'hr_gt': hr_rgb,
#             'hr_coord': hr_coord,
#             'lr_coord': hr_coord_full,
#             'mask': mask
#         }


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


# @register('sr-implicit-uniform-varied')
# class SRImplicitUniformVaried(Dataset):
#
#     def __init__(self, dataset, size_min, size_max=None,
#                  augment=False, gt_resize=None, sample_q=None):
#         self.dataset = dataset
#         self.size_min = size_min
#         if size_max is None:
#             size_max = size_min
#         self.size_max = size_max
#         self.augment = augment
#         self.gt_resize = gt_resize
#         self.sample_q = sample_q
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#
#         img_lr, img_lr_gt, mask, img_hr_gt = self.dataset[idx]
#         # p = idx / (len(self.dataset) - 1)
#         # w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
#         # img_hr_gt = resize_fn(img_hr_gt, w_hr)
#
#         lr_coord, lr_rgb = to_pixel_samples(img_lr_gt)
#         hr_coord_full, hr_rgb = to_pixel_samples(img_hr_gt)
#         hr_coord = hr_coord_full
#
#         if self.sample_q is not None:
#             sample_lst = np.random.choice(
#                 len(hr_coord_full), self.sample_q, replace=False)
#             hr_coord = hr_coord_full[sample_lst]
#             hr_rgb = hr_rgb[sample_lst]
#             # mask = mask[sample_lst]
#
#         return {
#             'inp': img_lr,
#             'lr_gt': img_lr_gt,
#             'hr_gt': hr_rgb,
#             'lr_coord': hr_coord_full,
#             'hr_coord': hr_coord,
#             'mask': mask
#         }


@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min=64, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)
        # return 10

    def __getitem__(self, idx):
        gt_img_feat, gt_img, masked_img_feat, mask, masked_img, c_mask = self.dataset[idx]


        crop_hr = gt_img
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        # gt_img_feat = gt_img_feat.view(512, -1).permute(1, 0)


        return {
            'gt_img_feat': gt_img_feat,
            'masked_img_feat': masked_img_feat,
            'hr_coord': hr_coord,
            'mask': mask,
            'gt_img': hr_rgb,
            'masked_img': masked_img,
            'c_mask': c_mask
        }