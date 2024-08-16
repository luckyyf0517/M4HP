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

from baselines.HuPR.models.networks import HuPRMultiTask
from baselines.HuPR.misc.losses import LossComputer
from baselines.HuPR.misc.plot import plotHumanPose


class MInterfaceMultitask(pl.LightningModule):
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
        self.confusion_matrix = np.zeros((self.cfg.MODEL.numClasses, self.cfg.MODEL.numClasses))
        
    def on_validation_start(self):
        self.save_preds = []
        self.confusion_matrix = np.zeros((self.cfg.MODEL.numClasses, self.cfg.MODEL.numClasses))
    
    def training_step(self, batch, batch_idx):
        # random_indices = np.random.choice(
        #     [i for i in range(self.cfg.TRAINING.batch_size * self.args.sampling_ratio)], 
        #     size=self.cfg.TRAINING.batch_size, replace=False)
        mmwave_cfg = batch['mmwave_cfg']
        if self.cfg.MODEL.recTarget == 'action': 
            labels = batch['action_label']
        elif self.cfg.MODEL.recTarget == 'person': 
            labels = batch['person_label']
        keypoints = batch['jointsGroup']
        VRDAEmaps_hori = batch['VRDAEmap_hori']
        VRDAEmaps_vert = batch['VRDAEmap_vert']
        heatmap, gcn_heatmap, cls_preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert, mmwave_cfg)
        loss, loss2, _, _ = self.loss_computer.compute_loss((heatmap, gcn_heatmap), keypoints)
        self.log('train_loss/loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('train_loss/loss2', loss2, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        loss_cls = self.lossCls(cls_preds * 10, labels) * 0.1
        self.log('train_loss/loss_cls', loss_cls, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        # print info
        num_iter = len(self.trainer.train_dataloader)
        self.print('epoch {0:04d}, iter {1:04d} / {2:04d} | TOTAL loss: {3:0>10.4f}'. \
            format(self.current_epoch, batch_idx, num_iter, loss.item()))
        return loss + loss_cls

    def validation_step(self, batch, batch_idx):
        bbox = batch['bbox']
        imageId = batch['imageId']
        mmwave_cfg = batch['mmwave_cfg']
        if self.cfg.MODEL.recTarget == 'action': 
            labels = batch['action_label']
        elif self.cfg.MODEL.recTarget == 'person': 
            labels = batch['person_label']
        keypoints = batch['jointsGroup']
        VRDAEmaps_hori = batch['VRDAEmap_hori']
        VRDAEmaps_vert = batch['VRDAEmap_vert']
        heatmap, gcn_heatmap, cls_preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert, mmwave_cfg)
        loss, loss2, preds2d, keypoints2d = self.loss_computer.compute_loss((heatmap, gcn_heatmap), keypoints)
        self.log('validation_loss/loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('validation_loss/loss2', loss2, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        loss_cls = self.lossCls(cls_preds * 10, labels) * 0.1
        self.log('validation_loss/loss_cls', loss_cls, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # print info
        num_iter = len(self.trainer.val_dataloaders)
        self.print('\033[93m' + 'epoch {0:04d}, iter {1:04d} / {2:04d} | TOTAL loss: {3:0>10.4f}'. \
            format(self.current_epoch, batch_idx, num_iter, loss.item()) + '\033[0m')
        # skip computing ap in validation
        # self.save_preds += self.save_keypoints(preds2d * self.cfg.DATASET.img_heatmap_ratio, bbox, imageId)
        self.update_confusion_matrix(cls_preds, labels)
        return loss + loss_cls

    def test_step(self, batch, batch_idx):
        bbox = batch['bbox']
        imageId = batch['imageId']
        mmwave_cfg = batch['mmwave_cfg']
        if self.cfg.MODEL.recTarget == 'action': 
            labels = batch['action_label']
        elif self.cfg.MODEL.recTarget == 'person': 
            labels = batch['person_label']
        keypoints = batch['jointsGroup']
        VRDAEmaps_hori = batch['VRDAEmap_hori']
        VRDAEmaps_vert = batch['VRDAEmap_vert']
        heatmap, gcn_heatmap, cls_preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert, mmwave_cfg)
        _, _, preds2d, keypoints2d = self.loss_computer.compute_loss((heatmap, gcn_heatmap), keypoints)
        # print info
        num_iter = len(self.trainer.test_dataloaders)
        self.print('\033[93m' + 'batch {0:04d} / {1:04d}'.format(batch_idx, num_iter) + '\033[0m')
        # draw with GT
        plotHumanPose(preds2d * self.cfg.DATASET.img_heatmap_ratio, imageIdx=imageId, cfg=self.cfg, vis_dir=os.path.join(self.args.vis_dir, self.args.version))
        # evaluation (by method in dataset)
        self.save_preds += self.save_keypoints(preds2d * self.cfg.DATASET.img_heatmap_ratio, bbox, imageId)
        self.update_confusion_matrix(cls_preds, labels)
    
    def on_train_end(self):
        return 
    
    def on_validation_epoch_end(self):
        self.compute_evaluate_metrics('val')
        return

    def on_test_epoch_end(self):
        self.compute_evaluate_metrics('test')
        return
        
    def compute_evaluate_metrics(self, phase='test'): 
        
        if phase == 'test': 
            self.write_keypoints(phase=phase)
            eval_dataset = self.trainer.test_dataloaders.dataset
            accAPs = eval_dataset.evaluateEach(loadDir=os.path.join('/root/log', self.args.version))
            for jointName, accAP in zip(self.cfg.DATASET.idx_to_joints, accAPs):
                self.log(phase + '_ap/' + jointName, accAP, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist_group=True)
            accAP = eval_dataset.evaluate(loadDir=os.path.join('/root/log', self.args.version))
            self.log(phase + '_ap/AP', accAP['AP'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist_group=True)
            self.log(phase + '_ap/Ap .5', accAP['AP .5'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist_group=True)
            self.log(phase + '_ap/Ap .75', accAP['AP .75'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist_group=True)
        
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
        
    def configure_optimizers(self):
        # Optimizer
        self.step_size = self.cfg.TRAINING.warmup_epoch
        # self.step_size = len(self.trainer.train_dataloader) * self.cfg.TRAINING.warmup_epoch
        if self.cfg.TRAINING.warmup_epoch == -1: 
            LR = self.cfg.TRAINING.lr  
        else: 
            LR = self.cfg.TRAINING.lr / (self.cfg.TRAINING.warmup_growth ** self.step_size)
        if self.cfg.TRAINING.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=LR, momentum=0.90, weight_decay=1e-4)
        elif self.cfg.TRAINING.optimizer == 'adam':  
            optimizer = optim.AdamW(self.model.parameters(), lr=LR, betas=(0.99, 0.999), weight_decay=1e-4)
        # Scheduler
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

    def lr_lambda(self, epoch):
        step = self.global_step
        if step % self.cfg.TRAINING.lr_decayIter == 0:
            if epoch < self.cfg.TRAINING.warmup_epoch:
                return self.cfg.TRAIxNING.warmup_growth
            else:
                return self.cfg.TRAINING.lr_decay
        else: 
            return 1

    def configure_loss(self):
        self.loss_computer = LossComputer(self.cfg, self.device)
        self.lossCls = nn.CrossEntropyLoss()
        return

    def load_model(self, cfg: dict) -> nn.Module:
        self.model = HuPRMultiTask(cfg)
        if self.cfg.MODEL.preLoad:
            # for hupr running task, load the pretrained model
            print('Loading model dict from ' + cfg.MODEL.weightPath)
            self.model.load_state_dict(torch.load(cfg.MODEL.weightPath)['model_state_dict'], strict=True)
            
    def write_keypoints(self, phase):
        predFile = os.path.join(self.cfg.DATASET.log_dir, self.args.version, f"{phase}_results.json")
        with open(predFile, 'w') as fp:
            json.dump(self.save_preds, fp, indent=4)
    
    def save_keypoints(self, preds, bbox, image_id, pred_heatmap=None):
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
            if pred_heatmap is not None:
                for kpts in range(self.cfg.DATASET.num_keypoints):
                    predsigma[j, kpts] = pred_heatmap[j, kpts].var().item() * self.heatmap_size
                block["sigma"] = predsigma[j].tolist()
            block_copy = block.copy()
            savePreds.append(block_copy)
        return savePreds
    
    def update_confusion_matrix(self, preds, labels):
        preds = torch.argmax(preds, dim=1)
        for i in range(len(preds)):
            self.confusion_matrix[labels[i], preds[i]] += 1