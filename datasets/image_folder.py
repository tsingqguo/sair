import os
import json
from PIL import Image
import cv2
import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from random import randrange
from datasets import register


def addmask(img):
    _, h, w = img.shape
    mask_path = './load/celebAHQ/mask64'
    mask_name = os.listdir(mask_path)

    n = randrange(9000)
  
    mask_file = os.path.join(mask_path, mask_name[n])
  
    size = 256
    mask = Image.open(mask_file).resize((size, size), Image.BICUBIC)
    mask1 = (np.array(mask) > 255 * 0.8).astype(np.uint8)

    masko = mask1
    if img.shape[1] != size:
        cat1 = None
        blur1 = None
        cat2 = None
        mask1 = None
    else:
        blur1 = img * (1 - mask1)
        mask1 = torch.from_numpy(mask1).unsqueeze(0).float()
        blur1 = np.ascontiguousarray(blur1) # 3 n n
        blur1 = torch.from_numpy(blur1).float() / 255
        cat1 = torch.cat([blur1,mask1], dim = 0) # 4 n n
        # cat1 = blur1

    masko = torch.from_numpy(masko).float() #
    masko = masko.view(-1, 1)
    cat2 = torch.from_numpy(img).float() / 255
    return cat1, cat2, masko

# @register('image-folder')
# class ImageFolder(Dataset):
#
#     def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
#                  repeat=1, cache='none'):
#         self.repeat = repeat
#         self.cache = cache
#
#         if split_file is None:
#             filenames = sorted(os.listdir(root_path))
#         else:
#             with open(split_file, 'r') as f:
#                 filenames = json.load(f)[split_key]
#         if first_k is not None:
#             filenames = filenames[:first_k]
#
#         self.files = []
#         for filename in filenames:
#             # filename = filename[2:]
#
#             file = os.path.join(root_path, filename)
#
#             if cache == 'none':
#                 self.files.append(file)
#
#             elif cache == 'bin':
#                 bin_root = os.path.join(os.path.dirname(root_path),
#                     '_bin_' + os.path.basename(root_path))
#                 if not os.path.exists(bin_root):
#                     os.mkdir(bin_root)
#                     print('mkdir', bin_root)
#                 bin_file = os.path.join(
#                     bin_root, filename.split('.')[0] + '.pkl')
#                 if not os.path.exists(bin_file):
#                     with open(bin_file, 'wb') as f:
#                         pickle.dump(imageio.imread(file), f)
#                     print('dump', bin_file)
#                 self.files.append(bin_file)
#
#             elif cache == 'in_memory':
#                 self.files.append(transforms.ToTensor()(
#                     Image.open(file).convert('RGB')))
#
#
#     def __len__(self):
#         return len(self.files) * self.repeat
#         # return 10
#     def __getitem__(self, idx):
#         x = self.files[idx % len(self.files)]
#
#         if self.cache == 'none':
#             return transforms.ToTensor()(Image.open(x).convert('RGB'))
#
#         elif self.cache == 'bin':
#
#
#             with open(x, 'rb') as f:
#                 x = pickle.load(f)
#             x = np.ascontiguousarray(x.transpose(2, 0, 1))
#             cat1, cat2, mask = addmask(x)
#
#             return cat1, cat2, mask
#
#         elif self.cache == 'in_memory':
#             return x
#
#
# @register('paired-image-folders')
# class PairedImageFolders(Dataset):
#
#     def __init__(self, root_path_1, root_path_2, **kwargs):
#         self.dataset_1 = ImageFolder(root_path_1, **kwargs)
#         self.dataset_2 = ImageFolder(root_path_2, **kwargs)
#
#     def __len__(self):
#         return len(self.dataset_1)
#
#     def __getitem__(self, idx):
#         return self.dataset_1[idx][0], self.dataset_1[idx][1], self.dataset_1[idx][2], self.dataset_2[idx][1]



@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        # if split_file is None:
        # filenames = sorted(os.listdir(root_path))
        # else:
        with open(split_file, 'r') as f:
            filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            # filename = filename[2:]
            file = os.path.join(root_path, filename)
            self.files.append(file)

        self.masked_feat_dir = './load/masked_img_feat'
        self.gt_feat_dir = './load/64_feat'
        self.mask_dir = './load/mask_64'
        self.gt_img_dir = './load/64'
        self.masked_img_dir = './load/masked_img'
        self.c_masks_dir = './load/CelebAMask'

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        x = x.split('/')[-1]
        x = x.split('.')[0]
   
        c_mask = Image.open(os.path.join(self.c_masks_dir, str(x) +'.png')).resize((256, 256), Image.NEAREST)
        c_mask = np.array(c_mask)
       

        mask = Image.open(os.path.join(self.mask_dir, str(x)+'.png')).resize((256,256))
        mask =  (np.array(mask) > 255 * 0.8).astype(np.uint8)


        mask_img_feat = np.load(os.path.join(self.masked_feat_dir, str(x)+'.npy'))
        mask_img_feat = torch.from_numpy(mask_img_feat)

        gt_img_feat = np.load(os.path.join(self.gt_feat_dir, str(x) + '.npy'))
        gt_img_feat = torch.from_numpy(gt_img_feat)
       

        gt_img = Image.open(os.path.join(self.gt_img_dir, str(x) + '.png')).resize((256,256)).convert('RGB')
        gt_img = np.array(gt_img).transpose(2, 0, 1)

        masked_img = gt_img * (1 - mask)

        gt_img = torch.from_numpy(gt_img) / 255
        masked_img = torch.from_numpy(masked_img) / 255


        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)

        return gt_img_feat, gt_img, mask_img_feat, mask, masked_img, c_mask




@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
       
    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx][0], self.dataset_1[idx][1], self.dataset_1[idx][2], self.dataset_1[idx][3], self.dataset_1[idx][4], self.dataset_1[idx][5]