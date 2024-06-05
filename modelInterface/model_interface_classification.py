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

from baselines.HuPR.models.networks import HuPRClassificationNet
from baselines.HuPR.misc.losses import LossComputer
from baselines.HuPR.misc.plot import plotHumanPose


class MInterfaceHuPRClassification(pl.LightningModule):
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
        self.confusion_matrix = np.zeros((self.cfg.MODEL.numClasses, self.cfg.MODEL.numClasses))
        
    def on_validation_start(self):
        self.confusion_matrix = np.zeros((self.cfg.MODEL.numClasses, self.cfg.MODEL.numClasses))
    
    def training_step(self, batch, batch_idx):
        # random_indices = np.random.choice(
        #     [i for i in range(self.cfg.TRAINING.batchSize * self.args.sampling_ratio)], 
        #     size=self.cfg.TRAINING.batchSize, replace=False)
        mmwave_cfg = batch['mmwave_cfg']
        if self.cfg.MODEL.recTarget == 'action': 
            labels = batch['action_label']
        elif self.cfg.MODEL.recTarget == 'person': 
            labels = batch['person_label']
        VRDAEmaps_hori = batch['VRDAEmap_hori']
        VRDAEmaps_vert = batch['VRDAEmap_vert']
        preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert, mmwave_cfg)
        loss = self.compute_loss(preds, labels)
        self.log('train_loss/loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        # print info
        num_iter = len(self.trainer.train_dataloader)
        self.print('epoch {0:04d}, iter {1:04d} / {2:04d} | TOTAL loss: {3:0>10.4f}'. \
            format(self.current_epoch, batch_idx, num_iter, loss.item()))
        return loss

    def validation_step(self, batch, batch_idx):
        mmwave_cfg = batch['mmwave_cfg']
        if self.cfg.MODEL.recTarget == 'action': 
            labels = batch['action_label']
        elif self.cfg.MODEL.recTarget == 'person': 
            labels = batch['person_label']
        VRDAEmaps_hori = batch['VRDAEmap_hori']
        VRDAEmaps_vert = batch['VRDAEmap_vert']
        preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert, mmwave_cfg)
        loss = self.compute_loss(preds, labels)
        self.log('validation_loss/loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # print info
        num_iter = len(self.trainer.val_dataloaders)
        self.print('\033[93m' + 'epoch {0:04d}, iter {1:04d} / {2:04d} | TOTAL loss: {3:0>10.4f}'. \
            format(self.current_epoch, batch_idx, num_iter, loss.item()) + '\033[0m')
        self.update_confusion_matrix(preds, labels)
        return loss

    def test_step(self, batch, batch_idx):
        imageId = batch['imageId']
        mmwave_cfg = batch['mmwave_cfg']
        if self.cfg.MODEL.recTarget == 'action': 
            labels = batch['action_label']
        elif self.cfg.MODEL.recTarget == 'person': 
            labels = batch['person_label']
        VRDAEmaps_hori = batch['VRDAEmap_hori']
        VRDAEmaps_vert = batch['VRDAEmap_vert']
        preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert, mmwave_cfg)
        # print info
        num_iter = len(self.trainer.test_dataloaders)
        self.print('\033[93m' + 'batch {0:04d} / {1:04d}'.format(batch_idx, num_iter) + '\033[0m')
        # evaluation (by method in dataset)
        self.update_confusion_matrix(preds, labels)
    
    def on_train_end(self):
        return 
    
    def on_validation_epoch_end(self):
        self.compute_evaluate_metrics()

    def on_test_epoch_end(self):
        self.compute_evaluate_metrics()
    
    def update_confusion_matrix(self, preds, labels):
        preds = torch.argmax(preds, dim=1)
        for i in range(len(preds)):
            self.confusion_matrix[labels[i], preds[i]] += 1
    
    def compute_evaluate_metrics(self): 
        TP = np.diag(self.confusion_matrix)
        FN = np.sum(self.confusion_matrix, axis=1) - TP
        FP = np.sum(self.confusion_matrix, axis=0) - TP
        smooth = 1e-8
        precision = TP / (TP + FP + smooth)
        recall = TP / (TP + FN + smooth)
        f1 = 2 * precision * recall / (precision + recall + smooth)
        
        self.log('precision', precision.mean(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('recall', recall.mean(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('f1', f1.mean(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        print('precision: ', precision.mean(), 'recall: ', recall.mean(), 'f1: ', f1.mean())
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
                return self.cfg.TRAIxNING.warmupGrowth
            else:
                return self.cfg.TRAINING.lrDecay
        else: 
            return 1
    
    def configure_loss(self):
        self.loss_fn = nn.CrossEntropyLoss()
    
    def compute_loss(self, preds, labels, t=10):
        return self.loss_fn(preds * t, labels)

    def load_model(self, cfg: dict) -> nn.Module:
        assert cfg.MODEL.runClassification is True, "This model is for classification task only."
        self.model = HuPRClassificationNet(cfg)
        if self.cfg.MODEL.preLoad:
            # for hupr running task, load the pretrained encoder only
            print('Loading model dict from ' + cfg.MODEL.weightPath)
            pretrained_hupr_model = torch.load(cfg.MODEL.weightPath)['state_dict']
            pretrained_hupr_model = {k.replace('moodel.', ''): v for k, v in pretrained_hupr_model.items()}
            self.model.load_state_dict(pretrained_hupr_model, strict=False)
            # # freeze encoder layers
            # for param in self.model.RAchirpNet.parameters():
            #     param.requires_grad = False
            # for param in self.model.REchirpNet.parameters():
            #     param.requires_grad = False
            # for param in self.model.RAradarEncoder.parameters():
            #     param.requires_grad = False
            # for param in self.model.REradarEncoder.parameters():
            #     param.requires_grad = False