import torch
import numpy as np
import torch.nn as nn
from math import sqrt, pi, log
import torch.nn.functional as F
from misc import get_max_preds, generate_target


class LossComputer():
    def __init__(self, cfg, device):
        self.device = device
        self.cfg = cfg
        self.num_group_frames = self.cfg.DATASET.num_group_frames
        self.num_keypoints = self.cfg.DATASET.num_keypoints
        self.heatmap_size = self.width = self.height = self.cfg.DATASET.heatmap_size
        self.imgSize = self.imgWidth = self.imgHeight = self.cfg.DATASET.crop_size
        self.loss_decay = self.cfg.TRAINING.loss_decay
        self.use_weight = self.cfg.TRAINING.use_weight
        self.alpha = 0.0
        self.beta = 1.0
        self.bce = nn.BCELoss()
            
    def compute_loss(self, preds, gt):
        b = gt.size(0)
        heatmaps = torch.zeros((b, self.num_keypoints, self.height, self.width))
        gtKpts = torch.zeros((b, self.num_keypoints, 2))
        for i in range(len(gt)):
            heatmap, gtKpt = generate_target(gt[i], self.num_keypoints, self.heatmap_size, self.imgSize)
            heatmaps[i, :] = torch.tensor(heatmap)
            gtKpts[i] = torch.tensor(gtKpt)
        preds, preds2 = preds      
        loss1 = self.computeBCESingleFrame(preds.view(-1, self.num_keypoints, self.height, self.width), heatmaps)
        preds = preds.permute(0, 2, 1, 3, 4).reshape(-1, self.num_keypoints, self.height, self.width)
        loss2 = self.computeBCESingleFrame(preds2.view(-1, self.num_keypoints, self.height, self.width), heatmaps)
        preds2 = preds2.permute(0, 2, 1, 3, 4).reshape(-1, self.num_keypoints, self.height, self.width)
        if self.alpha < 1.0:
            self.alpha += self.loss_decay
            self.beta -= self.loss_decay
        if self.loss_decay != -1:
            loss = self.alpha * loss1 + self.beta * loss2
        else:
            loss = loss1 + loss2
        pred2d, _ = get_max_preds(preds2.detach().cpu().numpy())
        gt2d, _ = get_max_preds(heatmaps.detach().cpu().numpy())
        return loss, loss2, pred2d, gt2d

    def computeBCESingleFrame(self, preds, gt):
        # loss = self.bce(preds, gt.to(preds.device))
        weight = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 5, 1, 3, 5]).view(14, 1, 1)
        if self.use_weight: 
            weight = weight.to(preds.device)
        else: 
            weight = None
        loss = F.binary_cross_entropy(preds, gt.to(preds.device), weight=weight)
        return loss