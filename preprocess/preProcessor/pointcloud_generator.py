import sys
sys.path.append('.')

import os
import time
import mpld3
import cupy as np
from tqdm import tqdm
from einops import rearrange, repeat
from matplotlib import pyplot as plt

from mmRadar import FMCWRadar 
from mmRadar.beamformer import CaponBeamformer, MUSICBeamformer
from .preprocessor import PreProcessor
from .utils import find_peak


class PointCloudGenerator(PreProcessor):
    def __init__(self, source_dir, source_seqs, target_dir): 
        super().__init__(source_dir, source_seqs, target_dir)
        self.radar: FMCWRadar = None
        
    def run_processing(self): 
        for seq_name in self.source_seqs:
            print('Processing', seq_name)
            mmwave_cfg, path_bin_hori, path_bin_vert, _ = self.load_folder(source_path_folder=os.path.join(self.source_dir, seq_name))        
            self.radar = FMCWRadar(mmwave_cfg)
            self.process_data(path_bin_hori, target_path_folder=os.path.join(self.target_dir, seq_name), seq_name=seq_name)

    def process_data(self, path_bin, target_path_folder=None, seq_name=None): 
        data_complex = self.radar.read_data(path_bin, complex=True)
        for idx_frame in tqdm(range(self.radar.num_frames)): 
            data_frame = data_complex[idx_frame]
            point_cloud = self.process_data_frame(data_frame)
            
    def process_data_frame(self, data_frame):
        range_result = self.radar.range_fft(data_frame)
        range_result = self.radar.remove_direct_component(range_result, axis=-1)
        doppler_result = self.radar.doppler_fft(range_result, axis=-1)
        aoa_input, energy_result, range_scale, veloc_scale = self.get_targets(doppler_result, top_size=128)
        x_vec, y_vec, z_vec = self.compute_naive_xyz(aoa_input)
        x, y, z = x_vec * range_scale, y_vec * range_scale, z_vec * range_scale
        point_cloud = np.concatenate([x, y, z, veloc_scale, energy_result, range_scale])
        point_cloud = np.reshape(point_cloud, (6, -1))
        point_cloud = point_cloud[:, y_vec != 0]
        return point_cloud
    
    def get_targets(self, doppler_result, top_size): 
        # add all attennas
        doppler_result_db = np.abs(doppler_result).sum(axis=1)  # sum first?
        doppler_result_db = np.log10(doppler_result_db)
        # filter out the bins which are too close or too far from radar
        doppler_result_db[:8, :] = 0
        doppler_result_db[-144:, :] = 0
        # filter out the bins lower than energy threshold
        filter_result = np.zeros_like(doppler_result_db)    # [num_range_bins, num_doppler_bins]
        energy_thre = np.sort(doppler_result_db.ravel())[-top_size - 1]
        filter_result[doppler_result_db > energy_thre] = True  
        # get range-doppler indices
        det_peaks_indices = np.argwhere(filter_result == True)
        range_scale = det_peaks_indices[:, 0]
        range_scale = range_scale.astype(np.float64) * self.radar.config.range_resolution
        veloc_scale = det_peaks_indices[:, 1]
        veloc_scale = veloc_scale - self.radar.num_doppler_bins // 2
        veloc_scale = veloc_scale.astype(np.float64) * self.radar.config.doppler_resolution
        # get aoa inputs (doppler value at top samples)
        energy_result = doppler_result_db[filter_result == True]
        # azimuth and elevation estimation
        doppler_result = rearrange(doppler_result, 'r a d -> a r d')
        aoa_input = doppler_result[:, filter_result == True] 
        return aoa_input, energy_result, range_scale, veloc_scale

    def compute_naive_xyz(self, aoa_input, method='fft'): 
         # split value
        num_rx = self.radar.config.num_rx
        azimuth_ant = aoa_input[0: num_rx * 2, :]
        elevation_ant = aoa_input[num_rx * 2: , :]
        # azimuth estimation
        azimuth_max, azimuth_peak = self.compute_phase_shift(azimuth_ant, method=method)
        wx = 2 * self.radar.config.angle_resolution * (azimuth_max - self.radar.num_angle_bins // 2)
        # elevation estimation
        _, elevation_peak = self.compute_phase_shift(elevation_ant, method=method)
        wz = np.angle(azimuth_peak * elevation_peak.conj() * np.exp(1j * 2 * wx))
        # get xyz coordinate
        x_vector = wx / np.pi
        z_vector = wz / np.pi    
        y_vector = 1 - x_vector ** 2 - z_vector ** 2
        x_vector[y_vector < 0] = 0
        z_vector[y_vector < 0] = 0
        y_vector[y_vector < 0] = 0
        y_vector = np.sqrt(y_vector)
        
        return x_vector, y_vector, z_vector
    
    def compute_phase_shift(self, data_ant, method='music'):
        if method == 'fft': 
            data_fft = np.fft.fft(data_ant, n=self.radar.num_angle_bins, axis=0)
            data_org = np.fft.fftshift(data_fft, axes=0)
            data_max, data_peak = find_peak(data_org)
        elif method == 'beamform': 
            beamformer = CaponBeamformer(num_steps=self.radar.num_angle_bins, num_antennas=self.radar.num_antenna)
            data_ant_ = data_ant.T
            _, bm_weight = beamformer.steering(data_ant_[:, :, np.newaxis])
            data_org = np.einsum('i j k, i k -> i j', np.conj(bm_weight).transpose(0, 2, 1), data_ant_).T
            data_max, data_peak = find_peak(data_org)
        elif method == 'music': 
            data_fft = np.fft.fft(data_ant, n=self.radar.num_angle_bins, axis=0)
            data_fft = np.fft.fftshift(data_fft, axes=0)
            beamformer = MUSICBeamformer(num_steps=self.radar.num_angle_bins, num_antennas=self.radar.num_antenna)
            data_ant_ = data_ant.T      
            power_spectrum = np.zeros_like(data_fft)
            for idx in range(data_ant_.shape[0]):  
                power_spectrum[:, idx] = beamformer.steering(data_ant_[idx, :, np.newaxis])
            data_max, data_peak = find_peak(power_spectrum, peak_source=data_fft)
        return data_max, data_peak
