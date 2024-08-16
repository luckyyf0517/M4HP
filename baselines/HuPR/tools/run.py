import os
import torch
import numpy as np
import torch.optim as optim
from models import HuPRNet
from misc import plotHumanPose
from datasets import getDataset
import torch.utils.data as data
import torch.nn.functional as F
from tools.base import BaseRunner


class Runner(BaseRunner):
    def __init__(self, args, cfg):
        super(Runner, self).__init__(args, cfg)    
        if not args.eval:
            self.trainSet = getDataset('train', cfg, args)
            self.trainLoader = data.DataLoader(self.trainSet,
                                  self.cfg.TRAINING.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.SETUP.num_workers)
        else:
            self.trainLoader = [0] # an empty loader
        self.testSet = getDataset('test' if args.eval else 'val', cfg, args)
        self.testLoader = data.DataLoader(self.testSet, 
                              self.cfg.TEST.batch_size,
                              shuffle=False,
                              num_workers=cfg.SETUP.num_workers)
        self.model = HuPRNet(self.cfg).to(self.device)
        self.step_size = len(self.trainLoader) * self.cfg.TRAINING.warmup_epoch
        LR = self.cfg.TRAINING.lr if self.cfg.TRAINING.warmup_epoch == -1 else self.cfg.TRAINING.lr / (self.cfg.TRAINING.warmup_growth ** self.step_size)
        self.initialize(LR)
        self.beta = 0.0
    
    def eval(self, visualization=True, epoch=-1):
        self.model.eval()
        loss_list = []
        self.logger.clear(len(self.testLoader.dataset))
        savePreds = []
        
        for idx, batch in enumerate(self.testLoader):
            keypoints = batch['jointsGroup']
            bbox = batch['bbox']
            imageId = batch['imageId']
            with torch.no_grad():
                VRDAEmaps_hori = batch['VRDAEmap_hori'].float().to(self.device)
                VRDAEmaps_vert = batch['VRDAEmap_vert'].float().to(self.device)
                preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert)
                loss, loss2, preds, gts = self.loss_computer.compute_loss(preds, keypoints)
                self.logger.display(loss, loss2, keypoints.size(0), epoch)
                if visualization:
                    plotHumanPose(preds*self.img_heatmap_ratio, self.cfg, 
                                  self.vis_dir, imageId, None)
                    # for drawing GT
                    plotHumanPose(gts * self.img_heatmap_ratio, self.cfg, 
                                  self.vis_dir, imageId, None, truth=True)
            self.save_keypoints(savePreds, preds*self.img_heatmap_ratio, bbox, imageId)
            loss_list.append(loss.item())
        self.write_keypoints(savePreds)
        if self.args.keypoints:
            accAP = self.testSet.evaluateEach(self.dir)
        accAP = self.testSet.evaluate(self.dir)
        return accAP

    def train(self):
        # init tensorboard
        writer = SummaryWriter()
        global_step = 0
        
        for epoch in range(self.start_epoch, self.cfg.TRAINING.epochs):
            self.model.train()
            loss_list = []
            self.logger.clear(len(self.trainLoader.dataset))
            for idxBatch, batch in enumerate(self.trainLoader):
                self.optimizer.zero_grad()
                keypoints = batch['jointsGroup']
                bbox = batch['bbox']
                VRDAEmaps_hori = batch['VRDAEmap_hori'].float().to(self.device)
                VRDAEmaps_vert = batch['VRDAEmap_vert'].float().to(self.device)
                preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert)
                loss, loss2, _, _ = self.loss_computer.compute_loss(preds, keypoints)
                writer.add_scalar('Loss/train', loss, global_step)
                loss.backward()
                self.optimizer.step()                    
                self.logger.display(loss, loss2, keypoints.size(0), epoch)
                if idxBatch % self.cfg.TRAINING.lr_decayIter == 0: #200 == 0:
                  self.adjustLR(epoch)
                loss_list.append(loss.item())
                global_step += 1
            # accAP = self.eval(visualization=False, epoch=epoch)
            self.saveModelWeight(epoch, sum(loss_list))
            self.saveLosslist(epoch, loss_list, 'train')