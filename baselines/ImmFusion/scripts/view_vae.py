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

from src.data_interface import DInterface
from src.model_interface import MInterface, mean_per_vertex_error, mean_per_joint_position_error, reconstruction_error
from run_immfusion_lightning import parse_args
from src.modeling.v_autoencoder import AutoEncoder

import mpld3
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from IPython import embed



    
def show_3d_points(points: torch.Tensor, joint=False, label=''): 
    # shape: [677, 3]
    points = points.cpu().detach().numpy()
    if joint: 
        points = points[:22, :] 
    else: 
        points = points[22:, :]
        
    perspectives = {'xy': [0, 1], 'yz': [1, 2], 'xz': [0, 2]}
    for i, pers in enumerate(perspectives):
        points_2d = points[:, perspectives[pers]]
        ax = plt.subplot(1, 3, i + 1)
        ax.scatter(points_2d[:, 0], points_2d[:, 1])
        ax.set_title('perspective: ' + pers)
        ax.set_aspect('equal')
        ax.axis('off')
                
        
if __name__ == '__main__': 
    args = parse_args()
    args.data_path = 'datasets/mmBody'
    args.model = 'DiffusionFusion'
    # args.model = 'AdaptiveFusion'
    args.per_gpu_train_batch_size = 10
    args.freeze_backbone = True
    args.train = False
    
    runner = MInterface(args=args).load_from_checkpoint('output/fusediffusion-v1/checkpoints/last.ckpt', map_location='cuda', strict=False)
    # runner = MInterface(args=args).load_from_checkpoint('output/immfusion/checkpoints/last.ckpt', map_location='cuda', strict=False)
    runner.to('cuda')
    data = DInterface(args=args)
    data.setup()
    batch = next(iter(data.test_dataloader()))
    for item in batch:
        if isinstance(batch[item], (tuple, dict)): 
            batch[item] = (i.to('cuda') for i in batch[item])
        else: 
            batch[item] = batch[item].to('cuda')

    runner.init_sampler()
    runner.smpl.eval()
    runner.model.eval()
    data_dict, meta_masks, batch_size = runner.prepare_batch(batch)
    
    gt_vertices_sub2 = data_dict['gt_vertices_sub2']
    gt_3d_joints = data_dict['gt_3d_joints']
    # concatinate template joints and template vertices, and then duplicate to batch size
    gt_vertices = torch.cat([gt_3d_joints, gt_vertices_sub2], dim=1)
    gt_vertices = gt_vertices.expand(batch_size, -1, -1)    # [b, 677, 3]
    
    embed()
    pred_dict, pred_3d_joints, pred_vertices_sub2, _, _ = \
        runner.model(args, data_dict, runner.smpl, runner.mesh_sampler, meta_masks=meta_masks, is_train=False)
    pred_vertices = torch.cat([pred_3d_joints, pred_vertices_sub2], dim=1)
    
    # reconstruct gt using vautoencoder
    z = runner.model.vae.encode(gt_vertices).sample()
    pred_vertices = runner.model.vae.decode(z)
    
    plt.clf()
    plt.figure(figsize=(16, 12))
    show_3d_points(pred_vertices[1], joint=False, label='pred')
    show_3d_points(gt_vertices[1], joint=False, label='gt')
    mpld3.show()
    
    error_vertices = mean_per_vertex_error(pred_vertices[:, 22:, :], gt_vertices[:, 22:, :])
    error_joints = mean_per_joint_position_error(pred_vertices[:, :22, :], gt_vertices[:, :22, :])
    