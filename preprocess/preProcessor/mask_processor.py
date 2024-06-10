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


class MaskProcessor(PreProcessor):
    def __init__(self, source_dir, target_dir, full_view=False):
        super().__init__(source_dir, target_dir)
        self.radar: FMCWRadar = None
        self.full_view = full_view
        
    def run_processing(self): 
        for seq_name in self.source_seqs[5: ]:
            print('Processing', seq_name)
            mmwave_cfg, path_bin_hori, path_bin_vert, _ = self.load_folder(
                source_path_folder=os.path.join(self.source_dir, seq_name), load_video=False)  
            
            path_3d_annot = os.path.join(self.source_dir, seq_name, 'kinect/skeleton3d.json')
            data_3d_annot = json.load(open(path_3d_annot, 'r'))
            
            self.asize = 64
            self.rsize = 64
            self.rstart = 26
            
            mmwave_cfg['num_angle_bins'] = self.asize
            self.radar = FMCWRadar(mmwave_cfg)
            
            self.process_data(path_bin_hori, path_bin_vert, data_3d_annot, target_path_folder=os.path.join(self.target_dir, seq_name))
    
    def process_data(self, path_bin_hori, path_bin_vert, annotation, target_path_folder=None): 
        # data_complex_hori = self.radar.read_data(path_bin_hori, complex=True)
        # data_complex_vert = self.radar.read_data(path_bin_vert, complex=True)
        
        for idx_frame in tqdm(range(self.radar.num_frames)): 
            annotation_frame = annotation['%04d' % idx_frame]

            if 'skeleton' not in annotation_frame:
                try_frame = int(idx_frame) + 1
                while int(try_frame) < len(annotation): 
                    annotation_frame = annotation['%04d' % try_frame]
                    if 'skeleton' in annotation_frame: 
                        break
                    try_frame += 1
                assert try_frame - int(idx_frame) < 30, 'No skeleton found for frame %s' % idx_frame
            
            # data_frame_hori = data_complex_hori[idx_frame]
            # data_frame_vert = data_complex_vert[idx_frame]
            # ra_spec = to_numpy(self.process_data_frame(data_frame_hori))[self.rstart: self.rstart + self.rsize, :]
            # re_spec = to_numpy(self.process_data_frame(data_frame_vert))[self.rstart: self.rstart + self.rsize, :]
            # self.save_data(ra_spec, folder_name='range_azimuth_numpy', idx_frame=idx_frame, target_dir=target_path_folder)
            # self.save_data(re_spec, folder_name='range_elevate_numpy', idx_frame=idx_frame, target_dir=target_path_folder)
            
            ra_proj, re_proj = self.get_projection(annotation_frame)
            ra_proj[1] -= self.rstart
            re_proj[1] -= self.rstart
            
            ra_mask = self.get_mask(np.zeros((64, 64)), ra_proj)
            re_mask = self.get_mask(np.zeros((64, 64)), re_proj)
            self.save_data(ra_mask, folder_name='range_azimuth_mask', idx_frame=idx_frame, target_dir=target_path_folder)
            self.save_data(re_mask, folder_name='range_elevate_mask', idx_frame=idx_frame, target_dir=target_path_folder)
           
            self.save_data(ra_mask, folder_name='range_azimuth_img', idx_frame=idx_frame, target_dir=target_path_folder, img=True)
            self.save_data(re_mask, folder_name='range_elevate_img', idx_frame=idx_frame, target_dir=target_path_folder, img=True)
        
    def process_data_frame(self, data_frame):
        radar_data_8rx, radar_data_4rx = self.radar.parse_data(data_frame)
        # Get range data
        radar_data_8rx = self.radar.remove_direct_component(radar_data_8rx, axis=0)
        radar_data_8rx = self.radar.range_fft(radar_data_8rx)
        # Get doppler data
        radar_spec = self.radar.get_spectrum_data(radar_data_8rx, 'beamform')
        return radar_spec
    
    def get_projection(self, annotation_frame): 
        coords = []
        joints = annotation_frame['skeleton']['joints']
        for joint in joints: 
            coords.append(list(joint['position'].values())[:3])
        coords = np.array(coords)   # x, y, z
        
        ra_proj = coords[:, [0, 2]]
        re_proj = coords[:, [1, 2]]
        
        def cart2pol(coords, res=False):
            x = coords[:, 1] / 1000
            y = coords[:, 0] / 1000
            rho = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            if res: 
                rho /= self.radar.config.range_resolution 
                rho = rho.astype(int)
                phi /= (np.pi / self.radar.num_angle_bins)
                phi += self.radar.num_angle_bins // 2
                phi = phi.astype(int)
            return np.vstack([phi, rho])    # [xs (angle), ys (range)]
        
        ra_proj = cart2pol(ra_proj, True)
        re_proj = cart2pol(re_proj, True)
        return ra_proj, re_proj
    
    def get_mask(self, spec, annot_points, sigma=1, thres=2): 
        # plot gaussian
        confmap = np.zeros_like(spec)
        width, height = spec.shape
        for i in range(32):
            x = annot_points[0, i]
            y = annot_points[1, i]
            for i in range(-3*sigma, 3*sigma+1):
                for j in range(-3*sigma, 3*sigma+1):
                    if 0 <= x+i < width and 0 <= y+j < height:
                        confmap[y+j, x+i] += np.exp(-(i**2 + j**2)/(2*sigma**2))
        mask = confmap > thres
        return mask
    
    def save_data(self, data, folder_name, idx_frame=0, target_dir=None, img=False): 
        folder_dir = os.path.join(target_dir, folder_name)
        os.makedirs(folder_dir, exist_ok=True)
        if not img: 
            np.save(os.path.join(folder_dir, "%09d.npy" % idx_frame), data)
        else: 
            plt.imshow(data)
            plt.axis('off')
            plt.savefig(os.path.join(folder_dir, "%09d.png" % idx_frame))
            plt.close()
    
    