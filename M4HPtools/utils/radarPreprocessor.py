import torch
import torch.nn as nn
import torch.nn.functional as F

class RadarPreprocessor(nn.Module):
    def __init__(self, cfg): 
        super().__init__()
        self.num_group_frames = cfg.DATASET.num_group_frames
        self.num_frames = cfg.DATASET.num_frames
        self.num_chirps = cfg.DATASET.num_chirps
        self.r = cfg.DATASET.num_range_bins
        self.w = cfg.DATASET.num_azimuth_bins
        self.h = cfg.DATASET.num_elevate_bins
        self.perform_doppler_fft = cfg.DATASET.perform_doppler_fft
        
    def forward(self, x, mmwave_cfg): 
        # x: [batch, frame, chirp, range, angle, elevation]
        x = self._preprocess_bin(x, mmwave_cfg)
        start_chirp = self.num_chirps//2 - self.num_frames//2
        end_chirp = self.num_chirps//2 + self.num_frames//2
        # select sub frame
        x = x[:, :, start_chirp: end_chirp]  
        x = torch.stack([x.real, x.imag], dim=2)
        return x
        
    def _preprocess_bin(self, radar_data, mmwave_cfg): 
        batch_size, _, num_chirps, _, _ = radar_data.shape
        # to save cuda memory
        proc_radar_data = torch.zeros(
            (batch_size, self.num_group_frames, self.num_chirps, self.r, self.w, self.h), 
            dtype=torch.complex64, device=radar_data.device)
        for b in range(batch_size): 
            radar_data_batch = radar_data[b]
            # all mmwave configs in one batch is same
            num_doppler_bins = self.num_chirps
            num_range_bins = self.r * 4 
            assert num_doppler_bins == num_chirps, 'num_doppler_bins should always be num_chirps'
            assert num_range_bins == 256, 'num_range_bins should always be 256'
            num_angle_bins = self.w 
            num_elevation_bins = self.h 
            range_resolution = mmwave_cfg['range_resolution'][b].item()
            # shape of radar_data: [frames, chirps, antennas, adcs]
            radar_data_8rx = radar_data_batch[:, :, 0: 8, :]
            radar_data_4rx = radar_data_batch[:, :, 8: 12, :]
            # remove dc (along doppler axis)
            radar_data_8rx -= torch.mean(radar_data_8rx, dim=1, keepdim=True)
            radar_data_4rx -= torch.mean(radar_data_4rx, dim=1, keepdim=True)
            # range and doppler fft (keeps shape)
            radar_data_8rx = torch.fft.fft(radar_data_8rx, dim=3, n=num_range_bins)   
            radar_data_4rx = torch.fft.fft(radar_data_4rx, dim=3, n=num_range_bins)   
            if self.perform_doppler_fft: 
                radar_data_8rx = torch.fft.fft(radar_data_8rx, dim=1)   
                radar_data_4rx = torch.fft.fft(radar_data_4rx, dim=1)   
            # padding to align
            radar_data_4rx = F.pad(radar_data_4rx, (0, 0, 2, 2))
            # merge elevation dimension, get [frames, chirps, antennas, elevations, adcs]
            radar_data_merge = torch.stack([radar_data_8rx, radar_data_4rx], axis=3) 
            # # elevation fft before, get: [frame, doppler, angle, elevation, range]
            # radar_data_merge[:, :, 2: 6, :, :] = torch.fft.fft(radar_data_merge[:, :, 2: 6, :, :], dim=3, n=num_elevation_bins)
            # angle fft
            radar_data_merge = torch.fft.fft(radar_data_merge, dim=2, n=num_angle_bins)
            # elevation fft after, get: [frame, doppler, angle, elevation, range]
            radar_data_merge = torch.fft.fft(radar_data_merge, dim=3, n=num_elevation_bins)
            # perform shift 
            if self.perform_doppler_fft: 
                radar_data_merge = torch.fft.fftshift(radar_data_merge, dim=(1, 2, 3))
            else: 
                radar_data_merge = torch.fft.fftshift(radar_data_merge, dim=(2, 3))
            # reshape to: [frame, doppler, range, angle, elevation]
            radar_data_slc = radar_data_merge.permute(0, 1, 4, 2, 3)
            # select specific range
            center_range_bin = int(2.5 / range_resolution)  # 26 ~ 90
            radar_data_slc = radar_data_slc[:, :, center_range_bin - 32: center_range_bin + 32, :, :]
            # flip at range, angle, elevation dimension
            radar_data_slc = torch.flip(radar_data_slc, dims=(1, 3, 4))
            proc_radar_data[b] = radar_data_slc
        return proc_radar_data
