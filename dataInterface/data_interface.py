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
from torch.utils.data import DataLoader, DistributedSampler, Sampler
import torch.distributed as dist


class DInterface(pl.LightningDataModule):

    def __init__(self, batch_size=16, num_workers=8, dataset=None, cfg=None, args=None):
        super().__init__()
        
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        self.cfg = cfg
        self.args = args

        self.load_data_module(dataset)

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_set = self.generate_dataset('train')
            self.val_set = self.generate_dataset('val')
        elif stage == 'test':
            self.test_set = self.generate_dataset('test')
        else: 
            raise ValueError(f"Invalid stage: {stage}")

    def generate_dataset(self, stage):
        return self.Dataset(phase=stage, cfg=self.cfg, args=self.args)

    def train_dataloader(self):
        # sampler = SubsetShuffleSampler(self.train_set.keyimageIds)
        return DataLoader(
            self.train_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers, 
            # sampler=sampler,
            persistent_workers=True, 
            pin_memory=True)

    def val_dataloader(self):
        # sampler = SubsetShuffleSampler(self.val_set.keyimageIds, shuffle=False)
        return DataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            # sampler=sampler,
            persistent_workers=True, 
            pin_memory=True)

    def test_dataloader(self):
        # sampler = SubsetShuffleSampler(self.test_set.keyimageIds, shuffle=False)
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            # sampler=sampler,
            pin_memory=True)

    def load_data_module(self, dataset):
        self.Dataset = dataset
        

# class SubsetShuffleSampler(DistributedSampler):
#     def __init__(self, image_ids, shuffle=True):
#         super().__init__(deepcopy(image_ids)) # compute self.total_size and self.num_replicas
#         random_idx_pair = [i for i in enumerate(deepcopy(image_ids))]
        
#         if shuffle: 
#             random.shuffle(random_idx_pair)
        
#         length_map = {}
#         for index, name in random_idx_pair: 
#             imageId = '%09d' % name
#             seq_name = imageId[: 4]
#             if seq_name not in length_map: 
#                 length_map[seq_name] = [index]
#             else: 
#                 length_map[seq_name].append(index)
                
#         random_idx = []
#         for seq_name in length_map: 
#             random_idx += length_map[seq_name]
#         self.indices = random_idx

#     def __iter__(self):
#         # subsample
#         if self.rank != self.num_replicas - 1:
#             indices = deepcopy(self.indices[self.rank * self.total_size// self.num_replicas: (self.rank + 1) * self.total_size// self.num_replicas])
#         else:
#             indices = self.indices[self.rank * self.total_size// self.num_replicas: ]
#         try:
#             assert len(self)==len(indices)
#         except:
#             raise ValueError(f"{len(self), len(indices), len(self.indices), self.rank}")
#         return iter(indices)

#     def __len__(self):
#         return self.total_size // self.num_replicas
    
