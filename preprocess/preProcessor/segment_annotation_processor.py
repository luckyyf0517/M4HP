import sys
sys.path.append('.')
sys.path.append('./baselines/HuPR/preprocessing')

import os
import cv2
import yaml
import time
import json
import mpld3
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import IPython

from mmRadar import FMCWRadar
from .preprocessor import PreProcessor
from .utils import to_numpy, plot_heatmap2D


class SegmentAnnotationProcessor(PreProcessor):
    def __init__(self, source_dir, target_dir, full_view=False):
        super().__init__(source_dir, target_dir)
        self.radar: FMCWRadar = None
        self.full_view = full_view
        
    def run_processing(self): 
        for seq_name in self.source_seqs:
            print('Processing', seq_name)
            mmwave_cfg, path_bin_hori, path_bin_vert, _ = self.load_folder(
                source_path_folder=os.path.join(self.source_dir, seq_name), load_video=False)  
            
            path_3d_annot = os.path.join(self.source_dir, seq_name, 'kinect/skeleton3d.json')
            data_3d_annot = json.load(open(path_3d_annot, 'r'))
            
            mmwave_cfg['num_angle_bins'] = 256
            self.radar = FMCWRadar(mmwave_cfg)
            
            self.process_data(path_bin_hori, path_bin_vert, data_3d_annot, target_path_folder=os.path.join(self.target_dir, seq_name))
    
    def process_data(self, path_bin_hori, path_bin_vert, annotation, target_path_folder=None): 
        data_complex_hori = self.radar.read_data(path_bin_hori, complex=True)
        data_complex_vert = self.radar.read_data(path_bin_vert, complex=True)
        data_out_vert = np.zeros((self.radar.num_frames, 16, 256, 256, 8), dtype=np.complex_)
        data_out_hori = np.zeros((self.radar.num_frames, 16, 256, 256, 8), dtype=np.complex_)
        
        for idx_frame in tqdm(range(self.radar.num_frames)): 
            annotation_frame = annotation['%04d' % idx_frame]
            data_frame_hori = data_complex_hori[idx_frame]
            data_frame_vert = data_complex_vert[idx_frame]
            ra_spec = to_numpy(self.process_data_frame(data_frame_hori))
            re_spec = to_numpy(self.process_data_frame(data_frame_vert))
            self.save_data(ra_spec, folder_name='range_azimuth_numpy', idx_frame=idx_frame, target_dir=target_path_folder)
            self.save_data(re_spec, folder_name='range_elevate_numpy', idx_frame=idx_frame, target_dir=target_path_folder)
            from IPython import embed; embed() 
        
    def process_data_frame(self, data_frame):
        radar_data_8rx, radar_data_4rx = self.radar.parse_data(data_frame)
        # Get range data
        radar_data_8rx = self.radar.remove_direct_component(radar_data_8rx, axis=0)
        radar_data_8rx = self.radar.range_fft(radar_data_8rx)
        # Get doppler data
        radar_spec = self.radar.get_spectrum_data(radar_data_8rx, 'beamform')
        return radar_spec
    
    def get_points(self, annotation_frame): 
        coords = []
        joints = annotation_frame['skeleton']['joints']
        for joint in joints: 
            coords.append(list(joint['position'].values())[:3])
        coords = np.array(coords)
        return coords
    
    def save_data(self, data, folder_name, idx_frame=0, target_dir=None): 
        folder_dir = os.path.join(target_dir, folder_name)
        os.makedirs(folder_dir, exist_ok=True)
        np.save(os.path.join(folder_dir, "%09d.npy" % idx_frame), data)
    
    