import torch
import torch.nn as nn
import torch.nn.functional as F
from models.chirp_networks import MNet
from models.layers import Encoder3D, MultiScaleCrossSelfAttentionPRGCN, MultiScaleCrossSelfAttentionClassification, MultiScaleCrossSelfAttentionMultitask
from einops import rearrange
from IPython import embed
from matplotlib import pyplot as plt
import mpld3


from M4HPtools.utils.radarPreprocessor import RadarPreprocessor 


class HuPRNet(nn.Module):
    def __init__(self, cfg):
        super(HuPRNet, self).__init__()
        self.numFrames = cfg.DATASET.num_frames
        self.num_filters = cfg.MODEL.num_filters
        self.rangeSize = cfg.DATASET.num_range_bins
        self.heatmap_size = cfg.DATASET.heatmap_size
        self.azimuthSize = cfg.DATASET.num_azimuth_bins
        self.elevationSize = cfg.DATASET.num_elevate_bins
        self.num_group_frames = cfg.DATASET.num_group_frames
        
        self.RAchirpNet = MNet(2, self.num_filters, self.numFrames)
        self.REchirpNet = MNet(2, self.num_filters, self.numFrames)
        self.RAradarEncoder = Encoder3D(cfg)
        self.REradarEncoder = Encoder3D(cfg)
        self.radarDecoder = MultiScaleCrossSelfAttentionPRGCN(cfg, batchnorm=False, activation=nn.PReLU)
        # perform preprocess
        self.PreProcessLayer = RadarPreprocessor(cfg)

    def forward_chirp(self, VRDAEmaps_hori, VRDAEmaps_vert):
        batch_size, frameSize, _, dopplerSize, rangeSize, angleSize, elevationSize = VRDAEmaps_hori.shape
        
        # Normalization at range-angle dim
        VRDAEmaps_hori = rearrange(VRDAEmaps_hori, 'b f c d w h a -> (b f c d) a w h')
        VRDAEmaps_vert = rearrange(VRDAEmaps_vert, 'b f c d w h a -> (b f c d) a w h')
        VRDAEmaps_hori = F.layer_norm(VRDAEmaps_hori, normalized_shape=[rangeSize, angleSize])
        VRDAEmaps_vert = F.layer_norm(VRDAEmaps_vert, normalized_shape=[rangeSize, angleSize])
        VRDAEmaps_hori = rearrange(VRDAEmaps_hori, '(b f c d) a w h -> b f d c w h a', b=batch_size, f=frameSize, d=dopplerSize, w=rangeSize)
        VRDAEmaps_vert = rearrange(VRDAEmaps_vert, '(b f c d) a w h -> b f d c w h a', b=batch_size, f=frameSize, d=dopplerSize, w=rangeSize)
        
        # Shrink elevation dimension
        VRDAmaps_hori = VRDAEmaps_hori.mean(dim=6)
        VRDAmaps_vert = VRDAEmaps_vert.mean(dim=6)

        RAmaps = self.RAchirpNet(VRDAmaps_hori.view(batch_size * self.num_group_frames, -1, self.numFrames, self.rangeSize, self.azimuthSize))
        RAmaps = RAmaps.squeeze(2).view(batch_size, self.num_group_frames, -1, self.rangeSize, self.azimuthSize).permute(0, 2, 1, 3, 4)
        REmaps = self.REchirpNet(VRDAmaps_vert.view(batch_size * self.num_group_frames, -1, self.numFrames, self.rangeSize, self.azimuthSize))
        REmaps = REmaps.squeeze(2).view(batch_size, self.num_group_frames, -1, self.rangeSize, self.azimuthSize).permute(0, 2, 1, 3, 4)
        
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
    

# class HuPRClassificationNet(nn.Module):
#     def __init__(self, cfg):
#         super(HuPRClassificationNet, self).__init__()
#         self.numFrames = cfg.DATASET.numFrames
#         self.num_filters = cfg.MODEL.num_filters
#         self.rangeSize = cfg.DATASET.rangeSize
#         self.heatmap_size = cfg.DATASET.heatmap_size
#         self.azimuthSize = cfg.DATASET.azimuthSize
#         self.elevationSize = cfg.DATASET.elevationSize
#         self.num_group_frames = cfg.DATASET.num_group_frames
#         self.RAchirpNet = MNet(2, self.num_filters, self.numFrames)
#         self.REchirpNet = MNet(2, self.num_filters, self.numFrames)
#         self.RAradarEncoder = Encoder3D(cfg)
#         self.REradarEncoder = Encoder3D(cfg)
        
#         # classification decoder
#         self.radarDecoder = MultiScaleCrossSelfAttentionClassification(cfg, batchnorm=False, activation=nn.PReLU)
        
#         # perform preprocess
#         self.PreProcessLayer = PreProcess(cfg)

#     def forward_chirp(self, VRDAEmaps_hori, VRDAEmaps_vert):
#         batch_size, frameSize, _, dopplerSize, rangeSize, angleSize, elevationSize = VRDAEmaps_hori.shape
        
#         # Normalization at range-angle dim
#         VRDAEmaps_hori = rearrange(VRDAEmaps_hori, 'b f c d w h a -> (b f c d) a w h')
#         VRDAEmaps_vert = rearrange(VRDAEmaps_vert, 'b f c d w h a -> (b f c d) a w h')
#         VRDAEmaps_hori = F.layer_norm(VRDAEmaps_hori, normalized_shape=[rangeSize, angleSize])
#         VRDAEmaps_vert = F.layer_norm(VRDAEmaps_vert, normalized_shape=[rangeSize, angleSize])
#         VRDAEmaps_hori = rearrange(VRDAEmaps_hori, '(b f c d) a w h -> b f d c w h a', b=batch_size, f=frameSize, d=dopplerSize, w=rangeSize)
#         VRDAEmaps_vert = rearrange(VRDAEmaps_vert, '(b f c d) a w h -> b f d c w h a', b=batch_size, f=frameSize, d=dopplerSize, w=rangeSize)
        
#         # Shrink elevation dimension
#         VRDAmaps_hori = VRDAEmaps_hori.mean(dim=6)
#         VRDAmaps_vert = VRDAEmaps_vert.mean(dim=6)

#         RAmaps = self.RAchirpNet(VRDAmaps_hori.view(batch_size * self.num_group_frames, -1, self.numFrames, self.rangeSize, self.azimuthSize))
#         RAmaps = RAmaps.squeeze(2).view(batch_size, self.num_group_frames, -1, self.rangeSize, self.azimuthSize).permute(0, 2, 1, 3, 4)
#         REmaps = self.REchirpNet(VRDAmaps_vert.view(batch_size * self.num_group_frames, -1, self.numFrames, self.rangeSize, self.azimuthSize))
#         REmaps = REmaps.squeeze(2).view(batch_size, self.num_group_frames, -1, self.rangeSize, self.azimuthSize).permute(0, 2, 1, 3, 4)
        
#         return RAmaps, REmaps
    
#     def forward(self, VRDAEmaps_hori, VRDAEmaps_vert, mmwave_cfg=None):
        
#         # perform preprocess
#         if mmwave_cfg is not None: 
#             VRDAEmaps_hori = self.PreProcessLayer(VRDAEmaps_hori, mmwave_cfg)
#             VRDAEmaps_vert = self.PreProcessLayer(VRDAEmaps_vert, mmwave_cfg)
        
#         RAmaps, REmaps = self.forward_chirp(VRDAEmaps_hori, VRDAEmaps_vert)
#         RAl1feat, RAl2feat, RAfeat = self.RAradarEncoder(RAmaps)
#         REl1feat, REl2feat, REfeat = self.REradarEncoder(REmaps)
#         preds = self.radarDecoder(RAl1feat, RAl2feat, RAfeat, REl1feat, REl2feat, REfeat)
#         return preds
    
    
# class HuPRMultiTask(nn.Module):
#     def __init__(self, cfg):
#         super(HuPRMultiTask, self).__init__()
#         self.numFrames = cfg.DATASET.numFrames
#         self.num_filters = cfg.MODEL.num_filters
#         self.rangeSize = cfg.DATASET.rangeSize
#         self.heatmap_size = cfg.DATASET.heatmap_size
#         self.azimuthSize = cfg.DATASET.azimuthSize
#         self.elevationSize = cfg.DATASET.elevationSize
#         self.num_group_frames = cfg.DATASET.num_group_frames
#         self.RAchirpNet = MNet(2, self.num_filters, self.numFrames)
#         self.REchirpNet = MNet(2, self.num_filters, self.numFrames)
#         self.RAradarEncoder = Encoder3D(cfg)
#         self.REradarEncoder = Encoder3D(cfg)
        
#         # classification decoder
#         self.multiDecoder = MultiScaleCrossSelfAttentionMultitask(cfg, batchnorm=False, activation=nn.PReLU)
        
#         # perform preprocess
#         self.PreProcessLayer = PreProcess(cfg)

#     def forward_chirp(self, VRDAEmaps_hori, VRDAEmaps_vert):
#         batch_size, frameSize, _, dopplerSize, rangeSize, angleSize, elevationSize = VRDAEmaps_hori.shape
        
#         # Normalization at range-angle dim
#         VRDAEmaps_hori = rearrange(VRDAEmaps_hori, 'b f c d w h a -> (b f c d) a w h')
#         VRDAEmaps_vert = rearrange(VRDAEmaps_vert, 'b f c d w h a -> (b f c d) a w h')
#         VRDAEmaps_hori = F.layer_norm(VRDAEmaps_hori, normalized_shape=[rangeSize, angleSize])
#         VRDAEmaps_vert = F.layer_norm(VRDAEmaps_vert, normalized_shape=[rangeSize, angleSize])
#         VRDAEmaps_hori = rearrange(VRDAEmaps_hori, '(b f c d) a w h -> b f d c w h a', b=batch_size, f=frameSize, d=dopplerSize, w=rangeSize)
#         VRDAEmaps_vert = rearrange(VRDAEmaps_vert, '(b f c d) a w h -> b f d c w h a', b=batch_size, f=frameSize, d=dopplerSize, w=rangeSize)
        
#         # Shrink elevation dimension
#         VRDAmaps_hori = VRDAEmaps_hori.mean(dim=6)
#         VRDAmaps_vert = VRDAEmaps_vert.mean(dim=6)

#         RAmaps = self.RAchirpNet(VRDAmaps_hori.view(batch_size * self.num_group_frames, -1, self.numFrames, self.rangeSize, self.azimuthSize))
#         RAmaps = RAmaps.squeeze(2).view(batch_size, self.num_group_frames, -1, self.rangeSize, self.azimuthSize).permute(0, 2, 1, 3, 4)
#         REmaps = self.REchirpNet(VRDAmaps_vert.view(batch_size * self.num_group_frames, -1, self.numFrames, self.rangeSize, self.azimuthSize))
#         REmaps = REmaps.squeeze(2).view(batch_size, self.num_group_frames, -1, self.rangeSize, self.azimuthSize).permute(0, 2, 1, 3, 4)
        
#         return RAmaps, REmaps
    
#     def forward(self, VRDAEmaps_hori, VRDAEmaps_vert, mmwave_cfg=None):
        
#         # perform preprocess
#         if mmwave_cfg is not None: 
#             VRDAEmaps_hori = self.PreProcessLayer(VRDAEmaps_hori, mmwave_cfg)
#             VRDAEmaps_vert = self.PreProcessLayer(VRDAEmaps_vert, mmwave_cfg)
        
#         RAmaps, REmaps = self.forward_chirp(VRDAEmaps_hori, VRDAEmaps_vert)
#         RAl1feat, RAl2feat, RAfeat = self.RAradarEncoder(RAmaps)
#         REl1feat, REl2feat, REfeat = self.REradarEncoder(REmaps)
        
#         heatmap, gcn_heatmap, cls_preds = self.multiDecoder(RAl1feat, RAl2feat, RAfeat, REl1feat, REl2feat, REfeat)
        
#         heatmap = torch.sigmoid(heatmap).unsqueeze(2)
#         return heatmap, gcn_heatmap, cls_preds