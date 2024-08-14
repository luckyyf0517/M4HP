import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN_layers(nn.Module):
    def __init__(self, in_features, out_features, num_keypoints, bias=True):
        super(GCN_layers, self).__init__()
        self.bias = bias
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features, num_keypoints))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, adj.to(input.device))
        output = torch.matmul(self.weight, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class PRGCN(nn.Module):
    def __init__(self, cfg, A):
        super(PRGCN, self).__init__()
        self.num_group_frames = cfg.DATASET.num_group_frames
        self.num_filters = cfg.MODEL.num_filters
        self.width = cfg.DATASET.heatmap_size
        self.height = cfg.DATASET.heatmap_size
        self.num_keypoints = cfg.DATASET.num_keypoints
        self.featureSize = (self.height//2) * (self.width//2)
        self.L1 = GCN_layers(self.featureSize, self.featureSize, self.num_keypoints)
        self.L2 = GCN_layers(self.featureSize, self.featureSize, self.num_keypoints)
        self.L3 = GCN_layers(self.featureSize, self.featureSize, self.num_keypoints)
        self.A = A
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def generate_node_feature(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x = x.reshape(-1, self.num_keypoints, self.featureSize).permute(0, 2, 1)
        return x

    def gcn_forward(self, x):
        #x: (B, num_filters, num_keypoints)
        x2 = self.relu(self.L1(x, self.A))
        x3 = self.relu(self.L2(x2, self.A))
        keypoints = self.L3(x3, self.A)
        return keypoints.permute(0, 2, 1)

    def forward(self, x):
        nodeFeat = self.generate_node_feature(x)
        heatmap = self.gcn_forward(nodeFeat).reshape(-1, self.num_keypoints, (self.height//2), (self.width//2))
        heatmap = F.interpolate(heatmap, scale_factor=2.0, mode='bilinear', align_corners=True)
        return torch.sigmoid(heatmap).unsqueeze(1)