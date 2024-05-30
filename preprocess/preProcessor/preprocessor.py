import sys
sys.path.append('.')

import os
import cv2
import yaml
import time
import cupy as cp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class PreProcessor(): 
    def __init__(self, source_dir, source_seqs, target_dir): 
        self.source_dir = source_dir
        self.source_seqs = source_seqs
        self.target_dir = target_dir
        
        self.radar = None
        
    def load_folder(self, source_path_folder, load_video=False): 
        print(os.path.join(source_path_folder, "radar_config.yaml"))
        with open(os.path.join(source_path_folder, "radar_config.yaml"), 'r') as f:
            mmwave_cfg = yaml.load(f, Loader=yaml.FullLoader)
        path_bin_hori = os.path.join(source_path_folder, "adc_data_hori.bin")
        path_bin_vert = os.path.join(source_path_folder, "adc_data_vert.bin")
        path_video = os.path.join(source_path_folder, "video.mp4") if load_video else None
        
        return mmwave_cfg, path_bin_hori, path_bin_vert, path_video
    
    def process_video(self, path_video, target_path_folder): 
        cap = cv2.VideoCapture(path_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps // 10)
        
        for idx_frame in tqdm(range(self.radar.num_frames)): 
            for _ in range(interval): 
                ret, frame = cap.read() 
            if ret: 
                folder_dir = os.path.join(target_path_folder, 'camera')
                os.makedirs(folder_dir, exist_ok=True)
                cv2.imwrite(os.path.join(folder_dir, "%09d.jpg" % idx_frame), frame)
                
    def run_processing(self): 
        pass
    
    def process_data(self, **kwargs): 
        pass
    
    def process_data_frame(self, **kwargs): 
        pass