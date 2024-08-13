import sys
sys.path.append('.')

import yaml
import argparse

import torch
import torch.utils
import torchvision
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module='scipy')

from diffusers import AutoencoderKL
from src.data_interface import DInterface
from src.model_interface import MInterface, visualize_mesh
from run_immfusion_lightning import parse_args
from IPython import embed
from src.utils.geometric_layers import orthographic_projection

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


if __name__ == '__main__': 
    args = parse_args()
    args.data_path = 'datasets/mmBody'
    args.model = 'DiffusionFusion'
    # args.model = 'AdaptiveFusion'
    args.per_gpu_train_batch_size = 1
    # args.freeze_backbone = True
    args.train = True
    
    runner = MInterface(args=args)#.load_from_checkpoint('output/fusediffusion/checkpoints/last.ckpt', map_location='cuda')
    runner.to('cuda')
    data = DInterface(args=args)
    data.setup()
    batch = next(iter(data.train_dataloader()))
    for item in batch:
        if isinstance(batch[item], (tuple, dict)): 
            batch[item] = (i.to('cuda') for i in batch[item])
        else: 
            batch[item] = batch[item].to('cuda')

    runner.init_sampler()
    runner.smpl.eval()
    runner.model.train()
    
    data_dict, meta_masks, batch_size = runner.prepare_batch(batch)
    embed()
    
    pred_dict, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices = \
        runner.model(args, data_dict, runner.smpl, runner.mesh_sampler, meta_masks=meta_masks, is_train=True)
        
    sum = (pred_vertices.sum() + pred_3d_joints.sum())
    sum.backward()

    for k, v in runner.model.named_parameters():
        if v.grad is None and v.requires_grad:
            print(k)
    
    # # Visualization
    # plt.clf()
    # gt_3d_joints = batch['gt_3d_joints']
    # gt_2d_joints = gt_3d_joints[0, :, [0, 2]]
    # gt_2d_joints = gt_2d_joints.cpu().detach().numpy()
    # plt.scatter(gt_2d_joints[:, 0], gt_2d_joints[:, 1])
    
    # pred_2d_joints = pred_3d_joints[0, :, [0, 2]]
    # pred_2d_joints = pred_2d_joints.cpu().detach().numpy()
    # plt.scatter(pred_2d_joints[:, 0], pred_2d_joints[:, 1])

    # plt.axis('equal')
    # mpld3.show()
    