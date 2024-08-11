import torch
import torch.nn as nn
import torch.nn.functional as F
from models.chirp_networks import MNet
from models.layers import Encoder3D, MultiScaleCrossSelfAttentionPRGCN, MultiScaleCrossSelfAttentionClassification, MultiScaleCrossSelfAttentionMultitask
from einops import rearrange
from IPython import embed
from matplotlib import pyplot as plt
import mpld3


class PreProcess(nn.Module):
    def __init__(self, cfg): 
        super().__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numFrames = cfg.DATASET.numFrames
        self.numChirps = cfg.DATASET.numChirps
        self.r = cfg.DATASET.rangeSize
        self.w = cfg.DATASET.azimuthSize
        self.h = cfg.DATASET.elevationSize
        
        self.performDopplerFFT = cfg.DATASET.performDopplerFFT
        
    def forward(self, x, mmwave_cfg): 
        # x: [batch, frame, chirp, range, angle, elevation]
        x = self._preprocess_bin(x, mmwave_cfg)
        startChirp = self.numChirps//2 - self.numFrames//2
        endChirp = self.numChirps//2 + self.numFrames//2
        # select sub frame
        x = x[:, :, startChirp: endChirp]  
        x = torch.stack([x.real, x.imag], dim=2)
        return x
        
    def _preprocess_bin(self, radar_data, mmwave_cfg): 
        batch_size, _, num_chirps, _, _ = radar_data.shape
        # to save cuda memory
        proc_radar_data = torch.zeros((batch_size, self.numGroupFrames, self.numChirps, self.r, self.w, self.h), dtype=torch.complex64, device=radar_data.device)
        for b in range(batch_size): 
            radar_data_batch = radar_data[b]
            # all mmwave configs in one batch is same
            num_doppler_bins = self.numChirps
            num_range_bins = self.r * 4 #mmwave_cfg['num_range_bins'][b].item()
            assert num_doppler_bins == num_chirps, 'num_doppler_bins should always be 64'
            assert num_range_bins == 256, 'num_range_bins should always be 256'
            num_angle_bins = self.w #mmwave_cfg['num_angle_bins'][b].item()
            num_elevation_bins = self.h #mmwave_cfg['num_elevation_bins'][b].item()
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
            if self.performDopplerFFT: 
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
            if self.performDopplerFFT: 
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


class HuPRNet(nn.Module):
    def __init__(self, cfg):
        super(HuPRNet, self).__init__()
        self.numFrames = cfg.DATASET.numFrames
        self.numFilters = cfg.MODEL.numFilters
        self.rangeSize = cfg.DATASET.rangeSize
        self.heatmapSize = cfg.DATASET.heatmapSize
        self.azimuthSize = cfg.DATASET.azimuthSize
        self.elevationSize = cfg.DATASET.elevationSize
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.RAchirpNet = MNet(2, self.numFilters, self.numFrames)
        self.REchirpNet = MNet(2, self.numFilters, self.numFrames)
        self.RAradarEncoder = Encoder3D(cfg)
        self.REradarEncoder = Encoder3D(cfg)
        self.radarDecoder = MultiScaleCrossSelfAttentionPRGCN(cfg, batchnorm=False, activation=nn.PReLU)
        
        # perform preprocess
        self.PreProcessLayer = PreProcess(cfg)

    def forward_chirp(self, VRDAEmaps_hori, VRDAEmaps_vert):
        batchSize, frameSize, _, dopplerSize, rangeSize, angleSize, elevationSize = VRDAEmaps_hori.shape
        
        # Normalization at range-angle dim
        VRDAEmaps_hori = rearrange(VRDAEmaps_hori, 'b f c d w h a -> (b f c d) a w h')
        VRDAEmaps_vert = rearrange(VRDAEmaps_vert, 'b f c d w h a -> (b f c d) a w h')
        VRDAEmaps_hori = F.layer_norm(VRDAEmaps_hori, normalized_shape=[rangeSize, angleSize])
        VRDAEmaps_vert = F.layer_norm(VRDAEmaps_vert, normalized_shape=[rangeSize, angleSize])
        VRDAEmaps_hori = rearrange(VRDAEmaps_hori, '(b f c d) a w h -> b f d c w h a', b=batchSize, f=frameSize, d=dopplerSize, w=rangeSize)
        VRDAEmaps_vert = rearrange(VRDAEmaps_vert, '(b f c d) a w h -> b f d c w h a', b=batchSize, f=frameSize, d=dopplerSize, w=rangeSize)
        
        # Shrink elevation dimension
        VRDAmaps_hori = VRDAEmaps_hori.mean(dim=6)
        VRDAmaps_vert = VRDAEmaps_vert.mean(dim=6)

        RAmaps = self.RAchirpNet(VRDAmaps_hori.view(batchSize * self.numGroupFrames, -1, self.numFrames, self.rangeSize, self.azimuthSize))
        RAmaps = RAmaps.squeeze(2).view(batchSize, self.numGroupFrames, -1, self.rangeSize, self.azimuthSize).permute(0, 2, 1, 3, 4)
        REmaps = self.REchirpNet(VRDAmaps_vert.view(batchSize * self.numGroupFrames, -1, self.numFrames, self.rangeSize, self.azimuthSize))
        REmaps = REmaps.squeeze(2).view(batchSize, self.numGroupFrames, -1, self.rangeSize, self.azimuthSize).permute(0, 2, 1, 3, 4)
        
        return RAmaps, REmaps
    
    def forward(self, VRDAEmaps_hori, VRDAEmaps_vert, mmwave_cfg=None):
        
        # perform preprocess
        if mmwave_cfg is not None: 
            VRDAEmaps_hori = self.PreProcessLayer(VRDAEmaps_hori, mmwave_cfg)
            VRDAEmaps_vert = self.PreProcessLayer(VRDAEmaps_vert, mmwave_cfg)
        
        RAmaps, REmaps = self.forward_chirp(VRDAEmaps_hori, VRDAEmaps_vert)
        RAl1feat, RAl2feat, RAfeat = self.RAradarEncoder(RAmaps)
        REl1feat, REl2feat, REfeat = self.REradarEncoder(REmaps)
        output, gcn_heatmap = self.radarDecoder(RAl1feat, RAl2feat, RAfeat, REl1feat, REl2feat, REfeat)
        heatmap = torch.sigmoid(output).unsqueeze(2)
        return heatmap, gcn_heatmap
    

class HuPRClassificationNet(nn.Module):
    def __init__(self, cfg):
        super(HuPRClassificationNet, self).__init__()
        self.numFrames = cfg.DATASET.numFrames
        self.numFilters = cfg.MODEL.numFilters
        self.rangeSize = cfg.DATASET.rangeSize
        self.heatmapSize = cfg.DATASET.heatmapSize
        self.azimuthSize = cfg.DATASET.azimuthSize
        self.elevationSize = cfg.DATASET.elevationSize
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.RAchirpNet = MNet(2, self.numFilters, self.numFrames)
        self.REchirpNet = MNet(2, self.numFilters, self.numFrames)
        self.RAradarEncoder = Encoder3D(cfg)
        self.REradarEncoder = Encoder3D(cfg)
        
        # classification decoder
        self.radarDecoder = MultiScaleCrossSelfAttentionClassification(cfg, batchnorm=False, activation=nn.PReLU)
        
        # perform preprocess
        self.PreProcessLayer = PreProcess(cfg)

    def forward_chirp(self, VRDAEmaps_hori, VRDAEmaps_vert):
        batchSize, frameSize, _, dopplerSize, rangeSize, angleSize, elevationSize = VRDAEmaps_hori.shape
        
        # Normalization at range-angle dim
        VRDAEmaps_hori = rearrange(VRDAEmaps_hori, 'b f c d w h a -> (b f c d) a w h')
        VRDAEmaps_vert = rearrange(VRDAEmaps_vert, 'b f c d w h a -> (b f c d) a w h')
        VRDAEmaps_hori = F.layer_norm(VRDAEmaps_hori, normalized_shape=[rangeSize, angleSize])
        VRDAEmaps_vert = F.layer_norm(VRDAEmaps_vert, normalized_shape=[rangeSize, angleSize])
        VRDAEmaps_hori = rearrange(VRDAEmaps_hori, '(b f c d) a w h -> b f d c w h a', b=batchSize, f=frameSize, d=dopplerSize, w=rangeSize)
        VRDAEmaps_vert = rearrange(VRDAEmaps_vert, '(b f c d) a w h -> b f d c w h a', b=batchSize, f=frameSize, d=dopplerSize, w=rangeSize)
        
        # Shrink elevation dimension
        VRDAmaps_hori = VRDAEmaps_hori.mean(dim=6)
        VRDAmaps_vert = VRDAEmaps_vert.mean(dim=6)

        RAmaps = self.RAchirpNet(VRDAmaps_hori.view(batchSize * self.numGroupFrames, -1, self.numFrames, self.rangeSize, self.azimuthSize))
        RAmaps = RAmaps.squeeze(2).view(batchSize, self.numGroupFrames, -1, self.rangeSize, self.azimuthSize).permute(0, 2, 1, 3, 4)
        REmaps = self.REchirpNet(VRDAmaps_vert.view(batchSize * self.numGroupFrames, -1, self.numFrames, self.rangeSize, self.azimuthSize))
        REmaps = REmaps.squeeze(2).view(batchSize, self.numGroupFrames, -1, self.rangeSize, self.azimuthSize).permute(0, 2, 1, 3, 4)
        
        return RAmaps, REmaps
    
    def forward(self, VRDAEmaps_hori, VRDAEmaps_vert, mmwave_cfg=None):
        
        # perform preprocess
        if mmwave_cfg is not None: 
            VRDAEmaps_hori = self.PreProcessLayer(VRDAEmaps_hori, mmwave_cfg)
            VRDAEmaps_vert = self.PreProcessLayer(VRDAEmaps_vert, mmwave_cfg)
        
        RAmaps, REmaps = self.forward_chirp(VRDAEmaps_hori, VRDAEmaps_vert)
        RAl1feat, RAl2feat, RAfeat = self.RAradarEncoder(RAmaps)
        REl1feat, REl2feat, REfeat = self.REradarEncoder(REmaps)
        preds = self.radarDecoder(RAl1feat, RAl2feat, RAfeat, REl1feat, REl2feat, REfeat)
        return preds
    
    
class HuPRMultiTask(nn.Module):
    def __init__(self, cfg):
        super(HuPRMultiTask, self).__init__()
        self.numFrames = cfg.DATASET.numFrames
        self.numFilters = cfg.MODEL.numFilters
        self.rangeSize = cfg.DATASET.rangeSize
        self.heatmapSize = cfg.DATASET.heatmapSize
        self.azimuthSize = cfg.DATASET.azimuthSize
        self.elevationSize = cfg.DATASET.elevationSize
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.RAchirpNet = MNet(2, self.numFilters, self.numFrames)
        self.REchirpNet = MNet(2, self.numFilters, self.numFrames)
        self.RAradarEncoder = Encoder3D(cfg)
        self.REradarEncoder = Encoder3D(cfg)
        
        # classification decoder
        self.multiDecoder = MultiScaleCrossSelfAttentionMultitask(cfg, batchnorm=False, activation=nn.PReLU)
        
        # perform preprocess
        self.PreProcessLayer = PreProcess(cfg)

    def forward_chirp(self, VRDAEmaps_hori, VRDAEmaps_vert):
        batchSize, frameSize, _, dopplerSize, rangeSize, angleSize, elevationSize = VRDAEmaps_hori.shape
        
        # Normalization at range-angle dim
        VRDAEmaps_hori = rearrange(VRDAEmaps_hori, 'b f c d w h a -> (b f c d) a w h')
        VRDAEmaps_vert = rearrange(VRDAEmaps_vert, 'b f c d w h a -> (b f c d) a w h')
        VRDAEmaps_hori = F.layer_norm(VRDAEmaps_hori, normalized_shape=[rangeSize, angleSize])
        VRDAEmaps_vert = F.layer_norm(VRDAEmaps_vert, normalized_shape=[rangeSize, angleSize])
        VRDAEmaps_hori = rearrange(VRDAEmaps_hori, '(b f c d) a w h -> b f d c w h a', b=batchSize, f=frameSize, d=dopplerSize, w=rangeSize)
        VRDAEmaps_vert = rearrange(VRDAEmaps_vert, '(b f c d) a w h -> b f d c w h a', b=batchSize, f=frameSize, d=dopplerSize, w=rangeSize)
        
        # Shrink elevation dimension
        VRDAmaps_hori = VRDAEmaps_hori.mean(dim=6)
        VRDAmaps_vert = VRDAEmaps_vert.mean(dim=6)

        RAmaps = self.RAchirpNet(VRDAmaps_hori.view(batchSize * self.numGroupFrames, -1, self.numFrames, self.rangeSize, self.azimuthSize))
        RAmaps = RAmaps.squeeze(2).view(batchSize, self.numGroupFrames, -1, self.rangeSize, self.azimuthSize).permute(0, 2, 1, 3, 4)
        REmaps = self.REchirpNet(VRDAmaps_vert.view(batchSize * self.numGroupFrames, -1, self.numFrames, self.rangeSize, self.azimuthSize))
        REmaps = REmaps.squeeze(2).view(batchSize, self.numGroupFrames, -1, self.rangeSize, self.azimuthSize).permute(0, 2, 1, 3, 4)
        
        return RAmaps, REmaps
    
    def forward(self, VRDAEmaps_hori, VRDAEmaps_vert, mmwave_cfg=None):
        
        # perform preprocess
        if mmwave_cfg is not None: 
            VRDAEmaps_hori = self.PreProcessLayer(VRDAEmaps_hori, mmwave_cfg)
            VRDAEmaps_vert = self.PreProcessLayer(VRDAEmaps_vert, mmwave_cfg)
        
        RAmaps, REmaps = self.forward_chirp(VRDAEmaps_hori, VRDAEmaps_vert)
        RAl1feat, RAl2feat, RAfeat = self.RAradarEncoder(RAmaps)
        REl1feat, REl2feat, REfeat = self.REradarEncoder(REmaps)
        
        heatmap, gcn_heatmap, cls_preds = self.multiDecoder(RAl1feat, RAl2feat, RAfeat, REl1feat, REl2feat, REfeat)
        
        heatmap = torch.sigmoid(heatmap).unsqueeze(2)
        return heatmap, gcn_heatmap, cls_preds