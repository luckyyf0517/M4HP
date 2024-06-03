import sys
sys.path.append('.')
sys.path.append('./baselines/HuPR/preprocessing')

import os
import cv2
import yaml
import time
import json
import mpld3
import cupy as cp
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
            
            self.radar = FMCWRadar(mmwave_cfg)
            self.radar.num_angle_bins = 128
            
            self.process_data(path_bin_hori, path_bin_vert, data_3d_annot, target_path_folder=os.path.join(self.target_dir, seq_name))
    
    def process_data(self, path_bin_hori, path_bin_vert, annotation, target_path_folder=None): 
        data_complex_hori = self.radar.read_data(path_bin_hori, complex=True)
        data_complex_vert = self.radar.read_data(path_bin_vert, complex=True)
        data_out_vert = np.zeros((self.radar.num_frames, 16, 64, 64, 8), dtype=np.complex_)
        data_out_hori = np.zeros((self.radar.num_frames, 16, 64, 64, 8), dtype=np.complex_)
        
        for idx_frame in tqdm(range(self.radar.num_frames)): 
            annotation_frame = annotation['%04d' % idx_frame]
            data_frame_hori = data_complex_hori[idx_frame]
            data_frame_vert = data_complex_vert[idx_frame]
            spec_frame_hori = to_numpy(self.process_data_frame(data_frame_hori))
            spec_frame_vert = to_numpy(self.process_data_frame(data_frame_vert))
            # self.save_data(cube_frame_hori, 'hori', idx_frame=idx_frame, target_dir=target_path_folder)
            # self.save_data(cube_frame_vert, 'vert', idx_frame=idx_frame, target_dir=target_path_folder)
            # if idx_frame % 20 == 0: 
            #     self.save_plot(cube_frame_hori, cube_frame_vert, idx_frame=idx_frame, seq_name=seq_name)
            data_out_hori[idx_frame, :, :, :, :] = cube_frame_hori
            data_out_vert[idx_frame, :, :, :, :] = cube_frame_vert
        self.save_all_data(data_out_hori, 'hori', target_dir=target_path_folder)
        self.save_all_data(data_out_vert, 'vert', target_dir=target_path_folder)
        
    def process_data_frame(self, data_frame):
        radar_data_8rx, radar_data_4rx = self.radar.parse_data(data_frame)
        # Get range data
        radar_data_8rx = self.radar.remove_direct_component(radar_data_8rx, axis=0)
        radar_data_8rx = self.radar.range_fft(radar_data_8rx)
        # Get doppler data
        radar_spec = self.radar.get_spectrum_data(radar_data_8rx, 'beamform')

        IPython.embed(); exit()
        
        return 
    
    def get_points(self, annotation_frame): 
        coords = []
        joints = annotation_frame['skeleton']['joints']
        for joint in joints: 
            coords.append(list(joint['position'].values())[:3])
        coords = np.array(coords)
        return coords
    
    def save_plot(self, data_hori, data_vert, idx_frame=0, seq_name=None): 
        plt.clf()
        plt.subplot(121)
        plot_heatmap2D(data_hori, axes=(1, 2), title='Range-Angle View')
        plt.subplot(122)
        plot_heatmap2D(data_vert, axes=(1, 2), title='Range-Elevation View', transpose=True)
        save_dir = '/root/viz/%s/heatmap' % seq_name
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, '%09d.png' % idx_frame))

    def save_data(self, data, view='hori', idx_frame=0, target_dir=None): 
        assert view in ['hori', 'vert'], 'Wrong view!!!'
        folder_dir = os.path.join(target_dir, view)
        os.makedirs(folder_dir, exist_ok=True)
        np.save(os.path.join(folder_dir, "%09d.npy" % idx_frame), data)
    
    def save_all_data(self, data, view='hori', target_dir=None): 
        assert view in ['hori', 'vert'], 'Wrong view!!!'
        np.save(os.path.join(target_dir, view + '.npy'), data)
    