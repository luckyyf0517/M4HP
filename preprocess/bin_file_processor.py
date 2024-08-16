import sys
sys.path.append('.')

import os
import json
import yaml
import glob
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

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

class AnnotationProcessor():
    def __init__(self):
        self.source_paths = sorted(glob.glob('M4HPDataset/Raw_files/*/*/*/mmwave/'))
        print('Found', len(self.source_paths), 'source paths')
    
    def run_processing(self): 
        for source_path in self.source_paths:
            print('Processing', source_path)
            cfg_path = os.path.join(source_path, "radar_config.yaml")
            with open(cfg_path, 'r') as f:
                mmwave_cfg = yaml.load(f, Loader=yaml.FullLoader)
            bin_hori = torch.from_numpy(np.fromfile(open(os.path.join(source_path, 'adc_data_hori.bin'), 'rb'), dtype=np.int16))
            bin_vert = torch.from_numpy(np.fromfile(open(os.path.join(source_path, 'adc_data_vert.bin'), 'rb'), dtype=np.int16))
            bin_hori = _reshape_bin(bin_hori, mmwave_cfg)
            bin_vert = _reshape_bin(bin_vert, mmwave_cfg)
            
            target_path = source_path.replace('Raw_files', 'Radar_files')
            
            for i in tqdm(range(mmwave_cfg['num_frames'])):
                target_path_hori = os.path.join(target_path, 'hori')
                target_path_vert = os.path.join(target_path, 'vert')
                os.makedirs(target_path_hori, exist_ok=True)
                os.makedirs(target_path_vert, exist_ok=True)
                torch.save(bin_hori[i].clone(), open(os.path.join(target_path_hori, '%04d.pt' % i), 'wb'))
                torch.save(bin_vert[i].clone(), open(os.path.join(target_path_vert, '%04d.pt' % i), 'wb'))

            # Copy the radar config file
            shutil.copy(cfg_path, os.path.join(target_path, 'radar_config.yaml'))

if __name__ == '__main__':
    processor = AnnotationProcessor()
    processor.run_processing()