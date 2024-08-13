from __future__ import absolute_import, division, print_function
import argparse
from copy import deepcopy
import os
import os.path as op
import code
import json
import time
import datetime
import torch
import torchvision.models as models
import gc
import numpy as np
import cv2
from openpyxl import load_workbook, Workbook
from torchvision.utils import make_grid
from src.datasets.utils import copy2cpu, crop_image, project_pcl, trans_mat_2_dict, INTRINSIC
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module='scipy')

from src.modeling.bert import BertConfig, Graphormer
import src.modeling.model as Models
from src.modeling.model import DiT_models
from src.modeling._smpl import SMPL, SMPLX, SMPLH36M, Mesh, SMPLXMesh
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.datasets.build import make_data_loader
from src.modeling.pointnet2.pointnet2_modules import PointnetSAModule
from src.utils.geometric_layers import orthographic_projection
import src.modeling.data.config as cfg
from src.utils.train_diffusion import requires_grad
from src.modeling.diffusion import create_diffusion


def build_model(args, logger): 
    # Mesh and SMPL utils
    smpl, mesh_sampler, max_position_embeddings = init_mesh(args, logger)
        
    start_epoch = 0
    if args.resume_checkpoint != None and args.resume_checkpoint!='None':
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _model = torch.load(os.path.join(args.resume_checkpoint, 'model.bin'), map_location='cpu')
        # for fine-tuning or resume training or inference, load weights from checkpoint
        logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
        start_epoch = int(args.resume_checkpoint.split('-')[1])
        # workaround approach to load sparse tensor in graph conv.
        states = torch.load(os.path.join(args.resume_checkpoint, 'state_dict.bin'), map_location='cpu')
        # del checkpoint_loaded
        _model.load_state_dict(states, strict=False)
        del states
        gc.collect()
        torch.cuda.empty_cache()
    else:        
        backbone = init_backbone(args, logger)
        # build end-to-end ImmFusion network (backbone + multi-layer Fusion Transformer)
        Model = getattr(Models, args.model)
        if args.model == 'AdaptiveDiffFusion': 
            diffusion = init_diffusion(args, logger)
            _model = Model(args, backbone, diffusion)
        else: 
            trans_encoder = init_transformer(args, logger, max_position_embeddings=max_position_embeddings)
            _model = Model(args, backbone, trans_encoder)
        # update configs to enable attention outputs
        if args.show_att:
            setattr(_model.trans_encoder[-1].config,'output_attentions', True)
            setattr(_model.trans_encoder[-1].config,'output_hidden_states', True)
            _model.trans_encoder[-1].bert.encoder.output_attentions = True
            _model.trans_encoder[-1].bert.encoder.output_hidden_states =  True
            for iter_layer in range(4):
                _model.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
            setattr(_model.trans_encoder[-1].config,'device', args.device)

    _model.to(args.device)
    return _model, smpl, mesh_sampler, start_epoch


def init_mesh(args, logger): 
    logger.info('Initializing mesh and smpl utils...')
    
    if args.mesh_type == 'smplx':
        smpl = SMPLX().to(args.device)
        mesh_sampler = SMPLXMesh()
        max_position_embeddings = 677
    elif args.mesh_type == 'smpl':
        smpl = SMPL().to(args.device)
        mesh_sampler = Mesh()
        max_position_embeddings = 455
    elif args.mesh_type == 'smplh36m':
        smpl = SMPLH36M().to(args.device)
        mesh_sampler = Mesh()
        max_position_embeddings = 445
    logger.info('Done.')
    return smpl, mesh_sampler, max_position_embeddings
    

def init_transformer(args, logger, max_position_embeddings=None): 
    logger.info('Initialing transformer-encoder blocks in a loop')
    
    trans_encoder = []
    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:] + [args.output_dim]  
    # which encoder block to have graph convs
    which_blk_graph = [int(item) for item in args.which_gcn.split(',')]
    
    for i in range(len(output_feat_dim)):
        config_class, model_class = BertConfig, Graphormer
        config = config_class.from_pretrained(args.config_name if args.config_name \
            else args.model_name_or_path)

        config.device = args.device
        config.output_attentions = False
        config.hidden_dropout_prob = args.drop_out
        config.img_feature_dim = input_feat_dim[i] 
        config.output_feature_dim = output_feat_dim[i]
        args.hidden_size = hidden_feat_dim[i]
        args.intermediate_size = int(args.hidden_size*args.interm_size_scale)
        config.max_position_embeddings = max_position_embeddings

        if which_blk_graph[i]==1:
            config.graph_conv = True
            logger.info("Add Graph Conv")
        else:
            config.graph_conv = False
        config.mesh_type = args.mesh_type

        # update model structure if specified in arguments
        update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

        for param in update_params:
            arg_param = getattr(args, param)
            config_param = getattr(config, param)
            if arg_param > 0 and arg_param != config_param:
                logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                setattr(config, param, arg_param)

        # init a transformer encoder and append it to a list
        assert config.hidden_size % config.num_attention_heads == 0
        model = model_class(config=config) 
        logger.info("Init model from scratch.")
        trans_encoder.append(model)
    
    trans_encoder = torch.nn.Sequential(*trans_encoder)
    total_params = sum(p.numel() for p in trans_encoder.parameters())
    logger.info('ImmFusion transformer total parameters: {}'.format(total_params))
    return trans_encoder    
        

def init_backbone(args, logger): 
    # init ImageNet pre-trained backbone model
    if args.arch=='hrnet':
        hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        img_backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
        logger.info('=> loading hrnet-v2-w40 model')
    elif args.arch=='hrnet-w64':
        hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        img_backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
        logger.info('=> loading hrnet-v2-w64 model')
    else:
        print("=> using pre-trained model '{}'".format(args.arch))
        # img_backbone = models.__dict__[args.arch](pretrained=True)
        img_backbone = models.__dict__[args.arch](pretrained=False)
        # remove the last fc layer
        img_backbone = torch.nn.Sequential(*list(img_backbone.children())[:-2])
        
    if args.model == 'PointWImageFeat':
        mlps = [2048, 4096, 4096, 2048]
    elif args.use_point_feat:
        mlps = [3, 128, 128, 1024, 1024, 2048]
    else:
        mlps = [0, 128, 128, 1024, 1024, 2048]
            
    radar_backbone = PointnetSAModule(npoint=args.num_clusters, radius=0.4, nsample=32, mlp=mlps.copy())
    depth_backbone = PointnetSAModule(npoint=49, radius=0.4, nsample=64, mlp=mlps.copy())
    backbone = dict(radar=radar_backbone, image=img_backbone, depth=depth_backbone)

    # backbone_total_params = 0
    # backbone_total_params += sum(p.numel() for p in backbone['radar'].parameters())
    # backbone_total_params += sum(p.numel() for p in backbone['image'].parameters())
    # logger.info('Backbone total parameters: {}'.format(backbone_total_params))
    return backbone


def init_diffusion(args, logger): 
    DiT = DiT_models['DiT-S/8']
    dit = DiT()
    ema = deepcopy(dit)
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    return {
        'diffusion': diffusion, 
        'dit': dit, 
        'ema': ema
    }