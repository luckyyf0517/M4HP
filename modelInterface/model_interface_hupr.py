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

import os
import time
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import importlib
import pytorch_lightning as pl

from baselines.HuPR.models.networks import HuPRNet
from baselines.HuPR.misc.losses import LossComputer
from baselines.HuPR.misc.plot import plotHumanPose


class MInterfaceHuPR(pl.LightningModule):
    def __init__(self, args, cfg, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.cfg = cfg
        self.args = args
        self.load_model(cfg)
        self.configure_loss()

    def forward(self, x):
        return self.model(x)
    
    def on_test_start(self):
        self.save_preds = []
        
    def on_validation_start(self):
        self.save_preds = []
    
    def training_step(self, batch, batch_idx):
        mmwave_cfg = batch['mmwave_cfg']
        keypoints = batch['jointsGroup']
        mmwave_bin_hori = batch['mmwave_bin_hori']
        mmwave_bin_vert = batch['mmwave_bin_vert']
        preds = self.model(mmwave_bin_hori, mmwave_bin_vert, mmwave_cfg)
        loss, loss2, _, _ = self.loss_computer.computeLoss(preds, keypoints)
        self.log('train_loss/loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('train_loss/loss2', loss2, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        # print info
        num_iter = len(self.trainer.train_dataloader)
        self.print('epoch {0:04d}, iter {1:04d} / {2:04d} | TOTAL loss: {3:0>10.4f}'. \
            format(self.current_epoch, batch_idx, num_iter, loss.item()))
        exit()
        return loss

    def validation_step(self, batch, batch_idx):
        bbox = batch['bbox']
        imageId = batch['imageId']
        mmwave_cfg = batch['mmwave_cfg']
        keypoints = batch['jointsGroup']
        VRDAEmaps_hori = batch['VRDAEmap_hori']
        VRDAEmaps_vert = batch['VRDAEmap_vert']
        preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert, mmwave_cfg)
        loss, loss2, preds2d, keypoints2d = self.lossComputer.computeLoss(preds, keypoints)
        self.log('validation_loss/loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('validation_loss/loss2', loss2, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # print info
        num_iter = len(self.trainer.val_dataloaders)
        self.print('\033[93m' + 'epoch {0:04d}, iter {1:04d} / {2:04d} | TOTAL loss: {3:0>10.4f}'. \
            format(self.current_epoch, batch_idx, num_iter, loss.item()) + '\033[0m')
        self.save_preds += self.saveKeypoints(preds2d * self.cfg.DATASET.imgHeatmapRatio, bbox, imageId)
        return loss

    def test_step(self, batch, batch_idx):
        bbox = batch['bbox']
        imageId = batch['imageId']
        mmwave_cfg = batch['mmwave_cfg']
        keypoints = batch['jointsGroup']
        VRDAEmaps_hori = batch['VRDAEmap_hori']
        VRDAEmaps_vert = batch['VRDAEmap_vert']
        preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert, mmwave_cfg)
        _, _, preds2d, keypoints2d = self.lossComputer.computeLoss(preds, keypoints)
        # print info
        num_iter = len(self.trainer.test_dataloaders)
        self.print('\033[93m' + 'batch {0:04d} / {1:04d}'.format(batch_idx, num_iter) + '\033[0m')
        # draw with GT
        plotHumanPose(preds2d * self.cfg.DATASET.imgHeatmapRatio, imageIdx=imageId, cfg=self.cfg, visDir=os.path.join(self.args.visDir, self.args.version))
        # evaluation (by method in dataset)
        self.save_preds += self.saveKeypoints(preds2d * self.cfg.DATASET.imgHeatmapRatio, bbox, imageId)
    
    def on_train_end(self):
        return 
    
    def on_validation_epoch_end(self):
        self.compute_evaluate_metrics('val')
        return

    def on_test_epoch_end(self):
        self.compute_evaluate_metrics('test')
        return
        
    def compute_evaluate_metrics(self, phase='test'): 
        self.writeKeypoints(phase=phase)
        
        if phase == 'test': 
            eval_dataset = self.trainer.test_dataloaders.dataset
        elif phase == 'val': 
            eval_dataset = self.trainer.val_dataloaders.dataset
        
        accAPs = eval_dataset.evaluateEach(loadDir=os.path.join('/root/log', self.args.version))
        for jointName, accAP in zip(self.cfg.DATASET.idxToJoints, accAPs):
            self.log(phase + '_ap/' + jointName, accAP, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist_group=True)
        
        accAP = eval_dataset.evaluate(loadDir=os.path.join('/root/log', self.args.version))
        self.log(phase + '_ap/AP', accAP['AP'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist_group=True)
        self.log(phase + '_ap/Ap .5', accAP['AP .5'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist_group=True)
        self.log(phase + '_ap/Ap .75', accAP['AP .75'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist_group=True)
        
    def configure_optimizers(self):
        # Optimizer
        self.stepSize = self.cfg.TRAINING.warmupEpoch
        # self.stepSize = len(self.trainer.train_dataloader) * self.cfg.TRAINING.warmupEpoch
        if self.cfg.TRAINING.warmupEpoch == -1: 
            LR = self.cfg.TRAINING.lr  
        else: 
            LR = self.cfg.TRAINING.lr / (self.cfg.TRAINING.warmupGrowth ** self.stepSize)
        if self.cfg.TRAINING.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=LR, momentum=0.90, weight_decay=1e-4)
        elif self.cfg.TRAINING.optimizer == 'adam':  
            optimizer = optim.AdamW(self.model.parameters(), lr=LR, betas=(0.99, 0.999), weight_decay=1e-4)
        # Scheduler
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

    def lr_lambda(self, epoch):
        step = self.global_step
        if step % self.cfg.TRAINING.lrDecayIter == 0:
            if epoch < self.cfg.TRAINING.warmupEpoch:
                return self.cfg.TRAINING.warmupGrowth
            else:
                return self.cfg.TRAINING.lrDecay
        else: 
            return 1

    def configure_loss(self):
        self.loss_computer = LossComputer(self.cfg, self.device)

    def load_model(self, cfg: dict) -> nn.Module:
        self.model = HuPRNet(cfg)
        # if self.cfg.MODEL.preLoad:
        #     # for hupr running task, load the pretrained model
        #     print('Loading model dict from ' + cfg.MODEL.weightPath)
        #     self.model.load_state_dict(torch.load(cfg.MODEL.weightPath)['model_state_dict'], strict=True)
            
    def writeKeypoints(self, phase):
        predFile = os.path.join(self.cfg.DATASET.log_dir, self.args.version, f"{phase}_results.json")
        with open(predFile, 'w') as fp:
            json.dump(self.save_preds, fp, indent=4)
    
    def saveKeypoints(self, preds, bbox, image_id, predHeatmap=None):
        savePreds = []
        visidx = np.ones((len(preds), self.cfg.DATASET.num_keypoints, 1))
        preds = np.concatenate((preds, visidx), axis=2)
        predsigma = np.zeros((len(preds), self.cfg.DATASET.num_keypoints))
        for j in range(len(preds)):
            block = {}
            block["category_id"] = 1
            block["image_id"] = int(image_id[j])
            block["score"] = 1.0
            block["keypoints"] = preds[j].reshape(self.cfg.DATASET.num_keypoints*3).tolist()
            if predHeatmap is not None:
                for kpts in range(self.cfg.DATASET.num_keypoints):
                    predsigma[j, kpts] = predHeatmap[j, kpts].var().item() * self.heatmap_size
                block["sigma"] = predsigma[j].tolist()
            block_copy = block.copy()
            savePreds.append(block_copy)
        return savePreds