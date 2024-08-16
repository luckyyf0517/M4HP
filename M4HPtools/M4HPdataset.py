import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.distributed as dist
import torchvision.transforms as transforms

import os
import sys
import yaml
import json
import time
import numpy as np
import glob
import multiprocessing
import PIL.Image as Image
from easydict import EasyDict as edict

from tqdm import tqdm

sys.path.append('.')


class M4HPSingleDataset(): 
    def __init__(self, phase, cfg, seq_name):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        self.phase = phase
        self.cfg = cfg.DATASET
        self.seq_name = seq_name
        
        self.load_raw_data = False  # load from bin file or eatracted files
        self.load_img = True
        self.load_mesh = False
        self.load_annotatoin = True
        
        self.data = {}
        self._load_dataset()
        
    def _load_dataset(self): 
        if self.load_raw_data: 
            _, _, mmwave_cfg = self._load_radar_data_from_bin(lazy_load=True)
        else: 
            _, _, mmwave_cfg = self._load_radar_data_from_folder()
        self.duration = mmwave_cfg['num_frames']
        self.data['mmwave_cfg'] = mmwave_cfg
        
        if self.load_img:
            img_list = self._load_img_list()
            self.data['img_list'] = img_list
            assert len(img_list) == self.duration, 'Number of images and radar frames do not match'

        if self.load_annotatoin: 
            joints2d, joints3d = self._load_annotation()
            self.data['joints2d'] = joints2d
            self.data['joints3d'] = joints3d
        
    def _load_radar_data_from_bin(self, lazy_load=True):
        folder_name, object_name, subseq_name = self.seq_name[0: 6], self.seq_name[6: 9], self.seq_name[9: 12] 
        radar_data_dir = os.path.join(self.cfg.root_dir, 'Raw_files', folder_name, object_name, subseq_name, 'mmwave')
        # load radar config
        with open(os.path.join(radar_data_dir, "radar_config.yaml"), 'r') as f:
            mmwave_cfg = yaml.load(f, Loader=yaml.FullLoader)
        if lazy_load:
            # to sppeds up the loading process, we only load the radar config
            bin_hori = None
            bin_vert = None
        else: 
            # load bin file to memory
            try: 
                bin_hori = self.loaded_bin['hori']
                bin_vert = self.loaded_bin['vert']
            except AttributeError:
                print(multiprocessing.current_process().name, 'Rank', dist.get_rank(), 'reloading', self.seq_name); 
                # print('reloading', self.seq_name); 
                bin_hori = torch.from_numpy(np.fromfile(open(os.path.join(radar_data_dir, 'adc_data_hori.bin'), 'rb'), dtype=np.int16))
                bin_vert = torch.from_numpy(np.fromfile(open(os.path.join(radar_data_dir, 'adc_data_vert.bin'), 'rb'), dtype=np.int16))
                bin_hori = self._reshape_bin(bin_hori, mmwave_cfg)
                bin_vert = self._reshape_bin(bin_vert, mmwave_cfg)
                self.loaded_bin = {'hori': bin_hori, 'vert': bin_vert}
        return bin_hori, bin_vert, mmwave_cfg
    
    def _load_radar_data_from_folder(self): 
        folder_name, object_name, subseq_name = self.seq_name[0: 6], self.seq_name[6: 9], self.seq_name[9: 12] 
        radar_data_dir = os.path.join(self.cfg.root_dir, 'Radar_files', folder_name, object_name, subseq_name, 'mmwave')
        # load radar config
        with open(os.path.join(radar_data_dir, "radar_config.yaml"), 'r') as f:
            mmwave_cfg = yaml.load(f, Loader=yaml.FullLoader)
            
        radar_files_hori = sorted(glob.glob(radar_data_dir + '/hori/*.pt'))  
        radar_files_vert = sorted(glob.glob(radar_data_dir + '/vert/*.pt'))
        return radar_files_hori, radar_files_vert, mmwave_cfg
        
    @staticmethod
    def _reshape_bin(bin_file, mmwave_cfg): 
        adc_shape = torch.tensor(mmwave_cfg['adc_shape'])
        file_size = torch.prod(adc_shape) * 2
        read_file_size = bin_file.shape[0]
        bin_file = F.pad(bin_file, (0, file_size - read_file_size), 'constant', 0)
        lvds0 = bin_file[0::2]
        lvds1 = bin_file[1::2]
        lvds0 = lvds0.reshape(*adc_shape)
        lvds1 = lvds1.reshape(*adc_shape)
        return lvds1 + 1j * lvds0
    
    def _load_img_list(self):
        folder_name, object_name, subseq_name = self.seq_name[0: 6], self.seq_name[6: 9], self.seq_name[9: 12] 
        img_dir = os.path.join(self.cfg.root_dir, 'Img_files', folder_name, object_name, subseq_name, self.cfg.img_type)
        img_list = sorted(glob.glob(img_dir + '/*.jpg'))
        return img_list
    
    def _load_annotation(self):
        folder_name, object_name, subseq_name = self.seq_name[0: 6], self.seq_name[6: 9], self.seq_name[9: 12] 
        annotaion_dir = os.path.join(self.cfg.root_dir, 'Annot_files', folder_name, object_name, subseq_name, 'annotation')
        joints2d_path = os.path.join(annotaion_dir, 'joints2d.json')
        joints3d_path = os.path.join(annotaion_dir, 'joints3d.json')
        joints2d = json.load(open(joints2d_path, 'r'))
        joints3d = json.load(open(joints3d_path, 'r'))
        return joints2d, joints3d
    
    def _select_group_frames(self, index):
        startFrame = index - self.cfg.num_group_frames//2 
        endFrame = index + self.cfg.num_group_frames//2
        if startFrame < 0: 
            endFrame -= startFrame
            startFrame = 0
        if endFrame > self.duration - 1: 
            startFrame -= (endFrame - self.duration + 1)
            endFrame = self.duration - 1
        return startFrame, endFrame
    
    def __getitem__(self, index): 
        item_dict = {'seq_name': self.seq_name, 'index': index}
        startFrame, endFrame = self._select_group_frames(index)
        
        if self.load_raw_data: 
            bin_hori, bin_vert, mmwave_cfg = self._load_radar_data_from_bin(lazy_load=False)
            mmwave_bin_hori = bin_hori[startFrame: endFrame]
            mmwave_bin_vert = bin_vert[startFrame: endFrame]
        else: 
            radar_files_hori, radar_files_vert, mmwave_cfg = self._load_radar_data_from_folder()
            mmwave_bin_hori = torch.stack([torch.load(radar_files_hori[i]) for i in range(startFrame, endFrame)])
            mmwave_bin_vert = torch.stack([torch.load(radar_files_vert[i]) for i in range(startFrame, endFrame)])
        item_dict['mmwave_bin_hori'] = mmwave_bin_hori
        item_dict['mmwave_bin_vert'] = mmwave_bin_vert
        item_dict['mmwave_cfg'] = mmwave_cfg
        
        if self.load_img: 
            image_path = self.data['img_list'][index]
            item_dict['image_path'] = image_path
        
        if self.load_annotatoin: 
            item_dict['joints2d'] = torch.tensor(self.data['joints2d'][index]['joints'])
            item_dict['joints3d'] = torch.tensor(self.data['joints3d'][index]['joints'])
            item_dict['bbox'] = torch.tensor(self.data['joints2d'][index]['bbox'])
        return item_dict        
    
    def __len__(self): 
        return self.duration
    
if __name__ == '__main__': 
    cfg = edict(yaml.safe_load(open('config/M4HPconfig.yaml', 'r')))
    train_dataset = data.ConcatDataset([
        M4HPSingleDataset('train', cfg, seq_name) for seq_name in cfg.DATASET.train_seqs])
    train_dataloader = data.DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=0)
    batch = next(iter(train_dataloader))
    
    from IPython import embed; embed()