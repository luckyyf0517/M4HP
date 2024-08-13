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
from copy import deepcopy
import pytorch_lightning as pl

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split

import src.datasets.fusion_dataset as Datasets
from src.utils.comm import get_world_size
from src.datasets.build import make_data_loader, filter_none_collate_fn


class DInterface(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.per_gpu_train_batch_size * get_world_size()

    def setup(self, stage=None):
        args = self.args
        dataset = self.generate_dataset(args)
        if args.train == True:
            if args.mix_data:
                datasets = [
                    dataset,
                    Datasets.CLIFFDataset(args, data_path='dataset/COCO'),
                    Datasets.CLIFFDataset(args, data_path='dataset/mpii'),
                ]
                dataset = ConcatDataset(datasets)
            if args.eval_test_dataset:
                eval_args = deepcopy(args)
                eval_args.train = False
                self.train_dataset = dataset
                self.val_dataset = self.generate_dataset(eval_args)
            else: 
                train_size = int(0.9 * len(dataset))
                eval_size = len(dataset) - train_size
                self.train_dataset, self.val_dataset = \
                    random_split(dataset, [train_size, eval_size])
            # # for quick debugging
            # self.train_dataset = Subset(self.train_dataset, range(0, len(self.train_dataset), 50))
            # self.val_dataset = Subset(self.val_dataset, range(0, len(self.val_dataset), 50))
        else: 
            self.test_dataset = dataset
            
    def generate_dataset(self, args):
        Dataset = getattr(Datasets, args.dataset)
        dataset = Dataset(args)
        return dataset

    def train_dataloader(self):
        args = self.args
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=args.num_workers, 
            pin_memory=True, collate_fn=filter_none_collate_fn, drop_last=True
        )

    def val_dataloader(self):
        args = self.args
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=args.num_workers,
            pin_memory=True, collate_fn=filter_none_collate_fn, drop_last=True
        )
    
    def test_dataloader(self):
        args = self.args
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=args.num_workers,
            pin_memory=True, collate_fn=filter_none_collate_fn, drop_last=True
        )

        