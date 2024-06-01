# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import importlib
import pytorch_lightning as pl

from baselines.HuPR.misc.losses import LossComputer
from baselines.HuPR.misc.plot import plotHumanPose


class MInterface(pl.LightningModule):
    def __init__(self, module: nn.Module, args, cfg, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.cfg = cfg
        self.args = args
        self.load_model(module, cfg)
        self.configure_loss()

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx):
        keypoints = batch['jointsGroup']
        VRDAEmaps_hori = batch['VRDAEmap_hori'].float()
        VRDAEmaps_vert = batch['VRDAEmap_vert'].float()
        preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert)
        loss, loss2, _, _ = self.lossComputer.computeLoss(preds, keypoints)
        self.log('train_loss/loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist_group=True)
        self.log('train_loss/loss2', loss2, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist_group=True)
        # print info
        num_iter = len(self.trainer.train_dataloader)
        self.print('epoch {0:04d}, iter {1:04d} / {2:04d} | TOTAL loss: {3:0>10.4f}'. \
            format(self.current_epoch, batch_idx, num_iter, loss.item()))
        return loss

    def validation_step(self, batch, batch_idx):
        imageId = batch['imageId']
        keypoints = batch['jointsGroup']
        VRDAEmaps_hori = batch['VRDAEmap_hori'].float().to(self.device)
        VRDAEmaps_vert = batch['VRDAEmap_vert'].float().to(self.device)
        preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert)
        loss, loss2, preds, gts = self.lossComputer.computeLoss(preds, keypoints)
        self.log('validation_loss/loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist_group=True)
        self.log('validation_loss/loss2', loss2, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist_group=True)
        # for drawing GT
        plotHumanPose(preds * self.cfg.DATASET.imgSize / self.cfg.DATASET.heatmapSize, self.cfg, self.visDir, imageId, None)
        # print info
        num_iter = len(self.trainer.val_dataloaders)
        self.print('\033[93m' + 'epoch {0:04d}, iter {1:04d} / {2:04d} | TOTAL loss: {3:0>10.4f}'. \
            format(self.current_epoch, batch_idx, num_iter, loss.item()) + '\033[0m')
        return loss

    def test_step(self, batch, batch_idx):
        imageId = batch['imageId']
        keypoints = batch['jointsGroup']
        VRDAEmaps_hori = batch['VRDAEmap_hori'].float().to(self.device)
        VRDAEmaps_vert = batch['VRDAEmap_vert'].float().to(self.device)
        preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert)
        # for drawing GT
        plotHumanPose(preds * self.cfg.DATASET.imgSize / self.cfg.DATASET.heatmapSize, self.cfg, self.visDir, imageId, None)
        # print info
        num_iter = len(self.trainer.test_dataloaders)
        self.print('\033[93m' + 'batch {0:04d} / {1:04d}'.format(batch_idx, num_iter) + '\033[0m')
        return
    
    def on_train_end(self):
        return 
    
    def on_validation_epoch_end(self):
        return

    def on_test_epoch_end(self):
        return 

    def configure_optimizers(self):
        # Optimizer
        self.stepSize = self.cfg.TRAINING.warmupEpoch
        # self.stepSize = len(self.trainer.train_dataloader) * self.cfg.TRAINING.warmupEpoch
        if self.cfg.TRAINING.warmupEpoch == -1: 
            LR = self.cfg.TRAINING.lr  
        else: 
            LR = self.cfg.TRAINING.lr / (self.cfg.TRAINING.warmupGrowth ** self.stepSize)
        if self.cfg.TRAINING.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
        elif self.cfg.TRAINING.optimizer == 'adam':  
            optimizer = optim.Adam(self.model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-4)
        # Scheduler
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

    def lr_lambda(self, epoch):
        step = self.global_step
        if step % self.cfg.TRAINING.lrDecayIter == 0:
            if epoch < self.cfg.TRAINING.warmupEpoch:
                return self.cfg.TRAIxNING.warmupGrowth
            else:
                return self.cfg.TRAINING.lrDecay
        else: 
            return 1

    def configure_loss(self):
        self.lossComputer = LossComputer(self.cfg, self.device)
        return

    def load_model(self, module: nn.Module, model_dict: dict, model_state_dict: dict=None) -> nn.Module:
        self.model: nn.Module = module(model_dict)
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)
            