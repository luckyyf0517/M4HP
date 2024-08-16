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
from baselines.HuPR.misc.coco import COCO
from baselines.HuPR.misc.cocoeval import COCOeval


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
        self.save_gts = {'annotations': [], 'images': [], 'categories': []}
        self.save_preds = []
        
    def on_validation_start(self):
        self.save_annots = []
        self.save_preds = []
    
    def training_step(self, batch, batch_idx):
        mmwave_cfg = batch['mmwave_cfg']
        keypoints = batch['joints2d']
        mmwave_bin_hori = batch['mmwave_bin_hori']
        mmwave_bin_vert = batch['mmwave_bin_vert']
        preds = self.model(mmwave_bin_hori, mmwave_bin_vert, mmwave_cfg)
        loss, loss2, _, _ = self.loss_computer.compute_loss(preds, keypoints)
        self.log('train_loss/loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('train_loss/loss2', loss2, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        # print info
        num_iter = len(self.trainer.train_dataloader)
        self.print('epoch {0:04d}, iter {1:04d} / {2:04d} | TOTAL loss: {3:0>10.4f}'. \
            format(self.current_epoch, batch_idx, num_iter, loss.item()))
        return loss

    def validation_step(self, batch, batch_idx):
        mmwave_cfg = batch['mmwave_cfg']
        keypoints = batch['joints2d']
        mmwave_bin_hori = batch['mmwave_bin_hori']
        mmwave_bin_vert = batch['mmwave_bin_vert']
        preds = self.model(mmwave_bin_hori, mmwave_bin_vert, mmwave_cfg)
        loss, loss2, preds2d, keypoints2d = self.loss_computer.compute_loss(preds, keypoints)
        self.log('validation_loss/loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('validation_loss/loss2', loss2, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # print info
        num_iter = len(self.trainer.val_dataloaders)
        self.print('\033[93m' + 'epoch {0:04d}, iter {1:04d} / {2:04d} | TOTAL loss: {3:0>10.4f}'. \
            format(self.current_epoch, batch_idx, num_iter, loss.item()) + '\033[0m')
        # self.save_preds += self.save_keypoints(preds2d * self.cfg.DATASET.img_heatmap_ratio, bbox, imageId)
        return loss

    def test_step(self, batch, batch_idx):
        bbox = batch['bbox']
        image_id = batch['index']
        mmwave_cfg = batch['mmwave_cfg']
        keypoints = batch['joints2d']
        mmwave_bin_hori = batch['mmwave_bin_hori']
        mmwave_bin_vert = batch['mmwave_bin_vert']
        preds = self.model(mmwave_bin_hori, mmwave_bin_vert, mmwave_cfg)
        _, _, preds2d, keypoints2d = self.loss_computer.compute_loss(preds, keypoints)
        # print info
        num_iter = len(self.trainer.test_dataloaders)
        self.print('\033[93m' + 'batch {0:04d} / {1:04d}'.format(batch_idx, num_iter) + '\033[0m')
        
        self.save_pred_keypoints(preds2d * self.cfg.DATASET.img_heatmap_ratio)
        self.save_pred_keypoints(keypoints2d)
    
    def on_train_end(self):
        return 
    
    def on_validation_epoch_end(self):
        self.compute_evaluate_metrics('val')

    def on_test_epoch_end(self):
        self.compute_evaluate_metrics('test')
        
    def compute_evaluate_metrics(self, phase='test'): 
        accAPs, accAP = self.coco_evaluation()
        for jointName, accAP_ in zip(self.cfg.DATASET.idx_to_joints, accAPs):
            self.log(phase + '_ap/' + jointName, accAP_, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist_group=True)
        self.log(phase + '_ap/AP', accAP['AP'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist_group=True)
        self.log(phase + '_ap/Ap .5', accAP['AP .5'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist_group=True)
        self.log(phase + '_ap/Ap .75', accAP['AP .75'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist_group=True)
        
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
                return self.cfg.TRAINING.warmup_growth
            else:
                return self.cfg.TRAINING.lr_decay
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
    
    def save_gt_keypoints(self, keypoints, bbox):
        visidx = np.ones((len(keypoints), 14, 1)) + 1.0 #2 is labeled and visible
        keypoints = np.concatenate((keypoints, visidx), axis=2)
        for j in range(len(keypoints)):
            block = {}
            block["num_keypoints"] = 14
            block["area"] = (bbox[j, 2] - bbox[j, 0]) * (bbox[j, 3] - bbox[j, 1])
            block["iscrowd"] = 0
            block["keypoints"] = keypoints[j].reshape(14*3).tolist()
            block["bbox"] = [bbox[j, 0], bbox[j, 1], bbox[j, 2] - bbox[j, 0], bbox[j, 3] - bbox[j, 1]]
            block["category_id"] = 1
            self.save_gts['annotations'].append(block.copy())            
            self.save_gts['images'].append({})            

    def save_pred_keypoints(self, preds):
        visidx = np.ones((len(preds), 14, 1))
        preds = np.concatenate((preds, visidx), axis=2)
        for j in range(len(preds)):
            block = {}
            block["category_id"] = 1
            block["score"] = 1.0
            block["keypoints"] = preds[j].reshape(14*3).tolist()
            self.save_preds.append(block.copy())
    
    def coco_evaluation(self): 
        # Generate GTs for COCO evaluation
        assert len(self.save_gts['annotations']) == len(self.save_preds), "The number of GTs and predictions should be the same"
        
        self.save_gts['categories'] = [{
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": [
                "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", 
                "L_Ankle", "Neck", "Head", "L_Shoulder", "L_Elbow", 
                "L_Wrist", "R_Shoulder", "R_Elbow", "R_Wrist"
            ],
            "skeleton": [
                [14,13],[13,12],[11,10],[10,9],[9,7],[12,9],[8,7],[7,1],[7,4],[6,5],[5,4],[3,2],[2,1]
            ]
        }]
        
        for i in range(len(self.save_preds)): 
            self.save_gts['annotations'][i]['image_id'] = i + 1
            self.save_gts['annotations'][i]['id'] = i + 1
            self.save_gts['images'][i]['id'] = i + 1
            self.save_preds[i]['image_id'] = i + 1
        
        coco_gt = COCO(self.save_gts)
        coco_dt = coco_gt.loadRes(self.save_preds)
        coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        
        # Evaluate for each keypoint
        accAPs = []
        for i in range(self.cfg.DATASET.num_keypoints):
            coco_eval.evaluate(i)
            coco_eval.accumulate()
            coco_eval.summarize()
            accAPs.append(coco_eval.stats[0])
            
        # Evaluate for all keypoints
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'AP .5', 'AP .75']
        accAP = {}
        for ind, name in enumerate(stats_names):
            accAP[name] = coco_eval.stats[ind]
        
        return accAPs, accAP