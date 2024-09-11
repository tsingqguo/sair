import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register
import models
from utils import make_coord
torch.cuda.empty_cache()
import os
from torchvision import transforms




@register('fir')
class FIR(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = True
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.l1_loss = nn.L1Loss()

        if imnet_spec is not None:
            imnet_in_dim = 65
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})



    def query_rgb(self, hr_coord):

        b, _, h, w = self.gt_feat.size()
        masked_feat_out = self.liif(self.masked_feat, hr_coord, self.imnet)
        masked_feat_out = masked_feat_out.view(b, h, w, -1).permute(0, 3, 1, 2)

        loss = 0
        return masked_feat_out, loss


    def forward(self, masked_feat, gt_feat, mask, hr_coord):

        self.gt_feat = gt_feat
        self.masked_feat = torch.cat([masked_feat, mask], dim=1)
        return self.query_rgb(hr_coord)


    def liif(self, feat, coord,  model):

        N, C, _, _ = feat.shape
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda()\
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        preds = []
        areas = []

        for vx in vx_lst:
            for vy in vy_lst:

                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)

                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)


                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                inp = torch.cat([q_feat, rel_coord], dim=-1)

                bs, q = coord.shape[:2]
                a = inp.view(bs * q, -1)
                pred = model(a).view(bs, q, -1)
                preds.append(pred)
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret
