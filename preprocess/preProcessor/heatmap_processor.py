import sys
sys.path.append('.')
sys.path.append('./baselines/HuPR/preprocessing')

import os
import cv2
import yaml
import time
import cupy as cp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from mmRadar import FMCWRadar
from .preprocessor import PreProcessor
from .utils import to_numpy, plot_heatmap2D


class HeatmapProcessor(PreProcessor):
    def __init__(self, source_dir, target_dir, full_view=False):
        super().__init__(source_dir, target_dir)
        self.radar: FMCWRadar = None
        self.full_view = full_view
        
    def run_processing(self): 
        for seq_name in self.source_seqs[15:]:
            seq_name = 'seq_0001'
            print('Processing', seq_name)
            mmwave_cfg, path_bin_hori, path_bin_vert, path_video = self.load_folder(
                source_path_folder=os.path.join(self.source_dir, seq_name), load_video=False)        
            self.radar = FMCWRadar(mmwave_cfg)
            if self.full_view: 
                self.radar.num_angle_bins = self.radar.num_range_bins
            self.process_data(path_bin_hori, path_bin_vert, target_path_folder=os.path.join(self.target_dir, seq_name), seq_name=seq_name)
            # if path_video is not None: 
            #     self.process_video(path_video, target_path_folder=os.path.join(self.target_dir, seq_name))
    
    def process_data(self, path_bin_hori, path_bin_vert, target_path_folder=None, seq_name=None): 
        data_complex_hori = self.radar.read_data(path_bin_hori, complex=True)
        # data_complex_vert = self.radar.read_data(path_bin_vert, complex=True)
        data_out_hori = np.zeros((self.radar.num_frames, 16, 64, 64, 8), dtype=np.complex_)
        # data_out_vert = np.zeros((self.radar.num_frames, 16, 64, 64, 8), dtype=np.complex_)
        for idx_frame in tqdm(range(self.radar.num_frames)): 
            data_frame_hori = data_complex_hori[idx_frame]
            # data_frame_vert = data_complex_vert[idx_frame]
            cube_frame_hori = to_numpy(self.process_data_frame(data_frame_hori))
            # cube_frame_vert = to_numpy(self.process_data_frame(data_frame_vert))
            # self.save_data(cube_frame_hori, 'hori', idx_frame=idx_frame, target_dir=target_path_folder)
            # self.save_data(cube_frame_vert, 'vert', idx_frame=idx_frame, target_dir=target_path_folder)
            # if idx_frame % 20 == 0: 
            #     self.save_plot(cube_frame_hori, cube_frame_vert, idx_frame=idx_frame, seq_name=seq_name)
            data_out_hori[idx_frame, :, :, :, :] = cube_frame_hori
            # data_out_vert[idx_frame, :, :, :, :] = cube_frame_vert
            exit()
        self.save_all_data(data_out_hori, 'hori', target_dir=target_path_folder)
        self.save_all_data(data_out_vert, 'vert', target_dir=target_path_folder)
        
    def process_data_frame(self, data_frame):
        radar_data_8rx, radar_data_4rx = self.radar.parse_data(data_frame)
        # Get range data
        radar_data_8rx = self.radar.remove_direct_component(radar_data_8rx, axis=0)
        radar_data_4rx = self.radar.remove_direct_component(radar_data_4rx, axis=0)
        radar_data_8rx = self.radar.range_fft(radar_data_8rx)
        radar_data_4rx = self.radar.range_fft(radar_data_4rx)
        # Get doppler data
        radar_data_8rx = self.radar.doppler_fft(radar_data_8rx, shift=False)
        radar_data_4rx = self.radar.doppler_fft(radar_data_4rx, shift=False)
        # Padding to align: [range, azimuth, elevation, doppler]
        radar_data_4rx = np.pad(radar_data_4rx, ((0, 0), (2, 2), (0, 0)))
        radar_data = np.stack([radar_data_8rx, radar_data_4rx], axis=2) 
        radar_data = np.pad(radar_data, ((0, 0), (0, 0), (0, self.radar.num_elevation_bins - 2), (0, 0)))
        # Get elevation data (along specific antenna)
        # radar_data[:, 2: 6,:, :] = self.radar.elevation_fft(radar_data[:, 2: 6,:, :], axis=2, shift=False)
        # Get angle data
        radar_data = self.radar.angle_fft(radar_data, axis=1, shift=False)
        radar_data = self.radar.elevation_fft(radar_data, axis=2, shift=False)
        # Shift the fft result
        radar_data = np.fft.fftshift(radar_data, axes=(1, 2, 3))    # [range, azimuth, elevation, doppler]
        if self.full_view: 
            radar_data_slc = radar_data
        else: 
            # Get the specific range
            center_range_bin = int(2.5 / self.radar.config.range_resolution)
            radar_data_slc = radar_data[center_range_bin - 32: center_range_bin + 32, :, :, :]
        # Select specific velocity
        radar_data_slc = radar_data_slc[:, :, :, self.radar.num_doppler_bins // 2 - 8: self.radar.num_doppler_bins // 2 + 8]
        # Flip at angle axis
        radar_data_slc = np.flip(radar_data_slc, axis=(0, 1, 2))
        return radar_data_slc.transpose(3, 0, 1, 2) 
    
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
    