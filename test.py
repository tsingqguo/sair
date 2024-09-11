import argparse
import os
import math
from functools import partial
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import datasets
import models
import utils
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torchvision
import lpips
from PIL import Image
from torchvision import transforms

def metric(gt, pre):

    # print(gt.size())
    pre = pre.clamp_(0, 1) * 255.0
    pre = pre.permute(0, 2, 3, 1)
    pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

    gt = gt.clamp_(0, 1) * 255.0
    gt = gt.permute(0, 2, 3, 1)
    gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

    psnr = min(100, compare_psnr(gt, pre))

    # print(gt.shape)
    ssim = compare_ssim(gt, pre, multichannel=True, data_range=255, channel_axis=2)

    return psnr, ssim


def batched_predict(model, inp, inp_gt, coord, cell, bsize):
    with torch.no_grad():
        p, _ = model.gen_feat(inp, inp_gt)

        # f1 = p[:, 10:13, :, :]
        # f1 = np.array(f1.squeeze(0).permute(1, 2, 0).detach().cpu())
        # # f1 = (f1 - np.min(f1)) / np.max(f1 - np.min(f1))
        # cv2.imwrite('f.png', f1 * 255)

        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()

    loss_fn_vgg = lpips.LPIPS(net='vgg')
    transf = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])



    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    psnr_list = []
    ssim_list = []
    l1_list = []
    lpips_list = []

    val_res = utils.Averager()
    pbar = tqdm(loader, leave=False, desc='val')
    for i, batch in enumerate(pbar):
        for k, v in batch.items():
            batch[k] = v.cuda()

        masked_img_feat = batch['masked_img_feat']
        gt_img_feat = batch['gt_img_feat']
        hr_coord = batch['hr_coord']
        mask = batch['mask']
        gt_img = batch['gt_img']
        masked_img = batch['masked_img']

        if eval_bsize is None:
            with torch.no_grad():
                pred, _ = model(masked_img_feat, gt_img_feat, mask, gt_img, masked_img, hr_coord)

                N = 1
                hw = 256
                pre = pred.view(N, hw, hw, 3).permute(0, 3, 1, 2)
                masked_img = pre

                pred, _ = model(masked_img_feat, gt_img_feat, mask, gt_img, masked_img, hr_coord)



         N = 1
         hw = 256
         gt = batch['gt_img'].view(N, hw, hw, 3).permute(0, 3, 1, 2)
         pre = pred.view(N, hw, hw, 3).permute(0, 3, 1, 2)

        
         psnr, ssim = metric(gt, pre)
         psnr_list.append(psnr)
         ssim_list.append(ssim)
         l1_loss = torch.nn.functional.l1_loss(gt, pre, reduction='mean').item()
         l1_list.append(l1_loss)
         pl = loss_fn_vgg(transf(pre[0].cpu()), transf(gt[0].cpu())).item()
         lpips_list.append(pl)
        
       
         print("psnr:{}/{}  ssim:{}/{} l1:{}/{}  lpips:{}/{}  {}".format(psnr, np.average(psnr_list),
                                                                         ssim, np.average(ssim_list),
                                                                         l1_loss, np.average(l1_list),
                                                                         pl, np.average(lpips_list),
                                                                         len(ssim_list)))


    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/test/test-celebAHQ-64-128.yaml')
    parser.add_argument('--model', default='./save/_train_celebAHQ-64-128_sair/epoch-best.pth')
    parser.add_argument('--gpu', default='0,1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True)


    print('result: {:.4f}'.format(res))
