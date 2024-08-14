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

import yaml
import random
import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, DistributedSampler, Sampler, ConcatDataset
import torch.distributed as dist

from M4HPtools.M4HPdataset import M4HPSingleDataset


class DInterface(pl.LightningDataModule):

    def __init__(self, batch_size=16, num_workers=8, cfg=None, args=None):
        super().__init__()
        
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        self.cfg = cfg
        self.args = args

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_set = self.generate_dataset('train')
            self.val_set = self.generate_dataset('val')
        elif stage == 'test':
            self.test_set = self.generate_dataset('test')
        else: 
            raise ValueError(f"Invalid stage: {stage}")

    def generate_dataset(self, stage):
        dataset = ConcatDataset([
            M4HPSingleDataset(phase=stage, cfg=self.cfg, seq_name=seq_name) 
            for seq_name in self.cfg.DATASET[stage + '_seqs']])
        # return self.Dataset(phase=stage, cfg=self.cfg, args=self.args)
        return dataset

    def train_dataloader(self):
        # sampler = SubsetShuffleSampler(self.train_set, shuffle=True)
        return DataLoader(
            self.train_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers, 
            shuffle=True, 
            # sampler=sampler,
            persistent_workers=True, 
            pin_memory=True)

    def val_dataloader(self):
        # sampler = SubsetShuffleSampler(self.val_set, shuffle=False)
        return DataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False, 
            # sampler=sampler,
            persistent_workers=True, 
            pin_memory=True)

    def test_dataloader(self):
        # sampler = SubsetShuffleSampler(self.test_set, shuffle=False)
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False, 
            # sampler=sampler,
            pin_memory=True)


# class SubsetShuffleSampler(DistributedSampler):
#     def __init__(self, dataset, shuffle=True):
#         super().__init__(deepcopy(dataset)) # compute self.total_size and self.num_replicas
#         self.shuffle = shuffle
#         self.indices = [i for i in range(len(dataset))]

#     def __iter__(self):
#         # split the indices for each gpu
#         if self.rank != self.num_replicas - 1:
#             indices = self.indices[self.rank * self.total_size // self.num_replicas: (self.rank + 1) * self.total_size// self.num_replicas]
#         else:
#             indices = self.indices[self.rank * self.total_size // self.num_replicas: ]
        
#         try:
#             assert len(self)==len(indices)
#         except:
#             raise ValueError(f"Length not corresponding: {len(self), len(indices), len(self.indices), self.rank}")
        
#         if self.shuffle: 
#             random.shuffle(indices)
        
#         return iter(indices)

    def __len__(self):
        return self.total_size // self.num_replicas
    
