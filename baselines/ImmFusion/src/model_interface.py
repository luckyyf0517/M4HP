import os
import gc
import time
import json
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from copy import deepcopy

import importlib
import pytorch_lightning as pl

from src.utils.miscellaneous import mkdir, set_seed
from src.utils.logger import setup_logger
from src.utils.comm import synchronize, is_main_process, get_rank, all_gather
import src.modeling.model as Models
from src.modeling.bert import BertConfig, Graphormer
from src.modeling.diffusion_transformer import DiT_models
from src.modeling._smpl import SMPL, SMPLX, SMPLH36M, Mesh, SMPLXMesh
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.modeling.pointnet2.pointnet2_modules import PointnetSAModule
from src.utils.geometric_layers import orthographic_projection
import src.modeling.data.config as cfg
from src.utils.train_diffusion import requires_grad
from src.modeling.diffusion import create_diffusion
from src.utils.renderer import Renderer, visualize_reconstruction, visualize_reconstruction_no_text
from src.utils.metric_logger import AverageMeter, EvalMetricsLogger
from src.utils.metric_pampjpe import reconstruction_error
from src.modeling.v_autoencoder import AutoEncoder


class MInterface(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.args = args
        self.load_model()
        self.configure_loss()

    def forward(self, x):
        raise NotImplementedError
    
    def on_train_start(self):
        # load mesh sampler 
        self.init_sampler()
        self.smpl.eval()
        self.model.train()
        
        # define loss function (criterion) 
        self.criterion_3d_keypoints = torch.nn.MSELoss(reduction='none')
        self.criterion_vertices = torch.nn.L1Loss()
        self.criterion_2d_keypoints = torch.nn.MSELoss(reduction='none')
        # renderer
        self.renderer = Renderer(faces=self.smpl.faces.cpu().numpy())
    
    def on_validation_start(self):
        # load mesh sampler 
        self.init_sampler()
        self.smpl.eval()
        self.model.eval()
        
        self.mPVE = AverageMeter()
        self.mPJPE = AverageMeter()
        self.PAmPJPE = AverageMeter()
        
    def on_test_start(self):
        return self.on_validation_start()
    
    def training_step(self, batch, batch_idx):
        args = self.args

        data_dict, meta_masks, batch_size = self.prepare_batch(batch)
     
        if args.pretrain_vae: 
            # ground truth data (normalized)
            gt_vertices_sub2 = data_dict['gt_vertices_sub2']
            gt_3d_joints = data_dict['gt_3d_joints']
            # concatinate template joints and template vertices, and then duplicate to batch size
            gt_vertices = torch.cat([gt_3d_joints, gt_vertices_sub2], dim=1)
            gt_vertices = gt_vertices.expand(batch_size, -1, -1)    # [b, 677, 3]
            kl_loss, reconst_loss = self.model(gt_vertices)
            self.print('Train epoch: {ep:02d}, iter: {iter:04d}/{iter_all:04d}, kl loss: {kl_loss:8.4f}, reconst_loss: {reconst_loss:8.4f}'.format(
                ep=self.current_epoch, iter=batch_idx, iter_all=len(self.trainer.train_dataloader), kl_loss=kl_loss, reconst_loss=reconst_loss))
            self.log('PreTrain/kl_loss', kl_loss.item(), on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log('PreTrain/reconst_loss', reconst_loss.item(), on_step=True, on_epoch=False, logger=True, sync_dist=True)
            return kl_loss + reconst_loss

        pred_dict, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices = \
            self.model(args, data_dict, self.smpl, self.mesh_sampler, meta_masks=meta_masks, is_train=True)
            
        # compute 3d joint loss  (where the joints are directly output from transformer)
        loss_3d_joints = keypoint_3d_loss(self.criterion_3d_keypoints, pred_3d_joints, data_dict['gt_3d_joints'], args.device)
        # compute 3d vertex loss 
        loss_vertices = (args.vloss_w_sub2 * vertices_loss(self.criterion_vertices, pred_vertices_sub2, data_dict['gt_vertices_sub2'], args.device) + \
                         args.vloss_w_sub * vertices_loss(self.criterion_vertices, pred_vertices_sub, data_dict['gt_vertices_sub'], args.device) + \
                         args.vloss_w_full * vertices_loss(self.criterion_vertices, pred_vertices, data_dict['gt_vertices'], args.device))
       
        loss = args.joints_loss_weight * loss_3d_joints + args.vertices_loss_weight * loss_vertices
        
        # compute diffusion loss
        if 'diffusion_loss' in pred_dict: 
            diffusion_loss = pred_dict['diffusion_loss'].mean()
            loss = args.diffusion_loss_weight * diffusion_loss
        else: 
            diffusion_loss = None
        
        # compute 2d joint loss
        # if args.joints_2d_loss:
        #     gt_2d_joints = data_dict['joints_2d'][args.joints_2d_loss].float().to(args.device)
        #     pred_2d_joints = orthographic_projection(pred_3d_joints, pred_dict['camera'])
        #     loss_2d_joints = keypoint_2d_loss(self.criterion_2d_keypoints, pred_2d_joints, gt_2d_joints)
        #     loss += args.vertices_loss_weight * loss_2d_joints
        #     # update metric logger and tensorboard
        #     self.log_loss_2djoints.update(loss_2d_joints.item(), batch_size)
        #     self.log('Train/2d_joint_loss', loss_2d_joints.item(), on_step=True, on_epoch=False, logger=True, sync_dist=True)
        
        # update tensorboard
        self.log('Train/3d_joint_loss', loss_3d_joints.item(), on_step=True, on_epoch=False, logger=True, sync_dist=False)
        self.log('Train/vertex_loss', loss_vertices.item(), on_step=True, on_epoch=False, logger=True, sync_dist=False)
        self.log('Train/total_loss', loss.item(), on_step=True, on_epoch=False, logger=True, sync_dist=False)
        
        # log text
        log_text = ''
        log_text += 'Train epoch: {ep:02d}, iter: {iter:04d}/{iter_all:04d}'.format(
            ep=self.current_epoch, iter=batch_idx, iter_all=len(self.trainer.train_dataloader))
        log_text += ' | loss: {:8.2f}, 3d joint loss: {:8.4f}, vertex loss: {:8.4f}'.format(loss.item(), loss_3d_joints.item(), loss_vertices.item())
        if diffusion_loss is not None: 
            self.log('Train/diffusion_loss', diffusion_loss.item(), on_step=True, on_epoch=False, logger=True, sync_dist=True)
            log_text += ', diffusion loss: {:8.4f}'.format(diffusion_loss.item())
        self.print(log_text)
        
        return loss

    def validation_step(self, batch, batch_idx):
        args = self.args
        
        data_dict, _, batch_size = self.prepare_batch(batch)
        
        if args.pretrain_vae: 
            # ground truth data (normalized)
            gt_vertices_sub2 = data_dict['gt_vertices_sub2']
            gt_3d_joints = data_dict['gt_3d_joints']
            # concatinate template joints and template vertices, and then duplicate to batch size
            gt_vertices = torch.cat([gt_3d_joints, gt_vertices_sub2], dim=1)
            gt_vertices = gt_vertices.expand(batch_size, -1, -1)    # [b, 677, 3]
            kl_loss, reconst_loss = self.model(gt_vertices)
            self.print('Valid iteration: {iter:04d}/{iter_all:04d}, kl loss: {kl_loss:8.4f}, reconst_loss: {reconst_loss:8.4f}'.format(
                iter=batch_idx, iter_all=len(self.trainer.val_dataloaders), kl_loss=kl_loss, reconst_loss=reconst_loss))
            self.log('PreValid/kl_loss', kl_loss.item(), on_step=False, on_epoch=True, logger=True, sync_dist=False)
            self.log('PreValid/reconst_loss', reconst_loss.item(), on_step=False, on_epoch=True, logger=True, sync_dist=False)
            return kl_loss + reconst_loss
        
        pred_dict, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices = \
            self.model(args, data_dict, self.smpl, self.mesh_sampler)
            
        gt_3d_joints = data_dict['gt_3d_joints']
        gt_vertices = data_dict['gt_vertices']
            
        pred_3d_pelvis = pred_3d_joints[:, 0, :]
        if args.dataset == 'Human36MDataset':
            gt_3d_joints = gt_3d_joints[:,cfg.J24_TO_J14,:]
            pred_3d_pelvis = self.smpl.get_h36m_joints(pred_vertices)[:,cfg.H36M_J17_NAME.index('Pelvis'),:]

        pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]
        pred_3d_joints = pred_3d_joints - pred_3d_pelvis[:, None, :]
        
        # measure errors
        error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices).mean()
        error_joints = mean_per_joint_position_error(pred_3d_joints, gt_3d_joints[:,:,:3]).mean()
        error_joints_pa = reconstruction_error(pred_3d_joints.cpu().numpy(), gt_3d_joints[:,:,:3].cpu().numpy(), reduction=None).mean()
        
        self.mPVE.update(error_vertices)
        self.mPJPE.update(error_joints)
        self.PAmPJPE.update(error_joints_pa)
        
        # log text
        log_text = ''
        if args.train: 
            self.log('Valid/mPVE', 1000 * float(error_vertices), on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('Valid/mPJPE', 1000 * float(error_joints), on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('Valid/PAmPJPE', 1000 * float(error_joints_pa), on_step=False, on_epoch=True, logger=True, sync_dist=True)
            log_text += 'Validation iteration: {iter}/{iter_all}'.format(
                iter=batch_idx, iter_all=len(self.trainer.val_dataloaders))
        else: 
            self.log('Test/mPVE', 1000 * float(error_vertices), on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('Test/mPJPE', 1000 * float(error_joints), on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('Test/PAmPJPE', 1000 * float(error_joints_pa), on_step=False, on_epoch=True, logger=True, sync_dist=True)
            log_text += 'Test iteration: {iter}/{iter_all}'.format(
                iter=batch_idx, iter_all=len(self.trainer.test_dataloaders))
        log_text += '| mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f} '.format(1000 * error_vertices, 1000 * error_joints, 1000 * error_joints_pa)
        self.print(log_text)
        
        # visualize 2d result
        # if True: 
        #     gt_2d_joints = data_dict['joints_2d'][args.joints_2d_loss].float()
        #     pred_2d_joints = orthographic_projection(pred_3d_joints, pred_dict['camera'])
        #     visual_imgs = visualize_mesh(self.renderer,
        #                                  data_dict['orig_img'][args.joints_2d_loss].detach(),
        #                                  data_dict['joints_2d'][args.joints_2d_loss][:,:,:2].detach(),
        #                                  pred_vertices.detach(), 
        #                                  pred_dict['camera'].detach(),
        #                                  pred_2d_joints.detach())
        #     visual_imgs = visual_imgs.permute(1,2,0).numpy()
        
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def on_train_end(self):
        return 
    
    def on_validation_epoch_end(self): 
        print(self.local_rank, 1000 * float(self.mPVE.avg), 1000 * float(self.mPJPE.avg), 1000 * float(self.PAmPJPE.avg))
        self.log('Recorded_mPVE', 1000 * float(self.mPVE.avg), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('Recorded_mPJPE', 1000 * float(self.mPJPE.avg), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('Recorded_PAmPJPE', 1000 * float(self.PAmPJPE.avg), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.print('Evaluation epoch done: {ep:02d}, mPVE: {mPVE:6.2f}, mPJPE: {mPJPE:6.2f}, PAmPJPE: {PAmPJPE:6.2f}' .format(
            ep=self.current_epoch, mPVE=1000 * self.mPVE.avg, mPJPE=1000 * self.mPJPE.avg, PAmPJPE=1000 * self.PAmPJPE.avg))
        # TODO: save error 
        # save_checkpoint(self.model, self.args, self.current_epoch, self.global_step)
    
    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()
  
    def configure_optimizers(self):
        args = self.args
        # Scheduler
        optimizer = torch.optim.Adam(params=list(self.model.parameters()),
                                     lr=args.lr,
                                     betas=(0.9, 0.999),
                                     weight_decay=0)
        def lr_lambda(epoch):
            return 0.1 ** (epoch // (self.args.num_train_epochs / 2.0))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

    def configure_loss(self):
        return
    
    def prepare_batch(self, batch): 
        batch_size = batch['joints_3d'].shape[0]
        # normalize gt joints
        gt_3d_joints = batch['joints_3d']
        gt_3d_pelvis = batch['root_pelvis'][:, None, :3]
        gt_3d_joints[:,:,:3] -= gt_3d_pelvis
        # generate simplified mesh
        gt_vertices = batch['vertices']
        gt_vertices_sub2 = self.mesh_sampler.downsample(gt_vertices, n1=0, n2=2)
        gt_vertices_sub = self.mesh_sampler.downsample(gt_vertices)
        # normalize gt based on smpl's pelvis
        gt_vertices_sub2 -= gt_3d_pelvis
        gt_vertices_sub -= gt_3d_pelvis
        gt_vertices -= gt_3d_pelvis
        # add gt into data batch
        batch['gt_vertices'] = gt_vertices
        batch['gt_vertices_sub'] = gt_vertices_sub
        batch['gt_vertices_sub2'] = gt_vertices_sub2
        batch['gt_3d_joints'] = gt_3d_joints
  
        # prepare masks for mask vertex/joint modeling
        joint_mask = batch['joint_mask']
        vert_mask = batch['vert_mask']
        joint_mask_ = joint_mask.expand(-1, -1, 2051)
        vert_mask_ = vert_mask.expand(-1, -1, 2051)
        meta_masks = torch.cat([joint_mask_, vert_mask_], dim=1)
        
        return batch, meta_masks, batch_size

    def load_model(self) -> nn.Module:
        args = self.args
        
        # load model
        self.init_mesh()
        self.mesh_sampler = None    # wait for mesh_sampler to be initialized
        
        if args.model == 'DiffusionFusion': 
            if not args.pretrain_vae: 
                self.model = self.init_model()
            else: 
                self.model = self.init_vae(args)
        else: 
            self.model = self.init_model()

    def init_mesh(self):
        args = self.args
        
        # load mesh and smpl utils
        smpl_path = 'src/modeling/data/%s_model.pth' % args.mesh_type
        if not os.path.exists(smpl_path): 
            smpl_dict = {'smpl': SMPL, 'smplh36m': SMPLH36M, 'smplx': SMPLX}
            smpl = smpl_dict[args.mesh_type]()
            torch.save(smpl, smpl_path)
        else: 
            smpl = torch.load(smpl_path)
        self.smpl = smpl
        self.render = Renderer(faces=smpl.faces.cpu().numpy())
    
    def init_sampler(self):
        args = self.args
        
        # mesh sampler should be initialized after distributed initialization
        if self.mesh_sampler is None: 
            mesh_sampler_dict = {'smpl': Mesh, 'smplh36m': Mesh, 'smplx': SMPLXMesh}
            mesh_sampler = mesh_sampler_dict[args.mesh_type]()
            self.mesh_sampler = mesh_sampler
    
    def init_model(self) -> nn.Module:        
        args = self.args
        
        # Load model
        backbone = self.init_backbone(args)
        # build end-to-end ImmFusion network (backbone + multi-layer Fusion Transformer)
        trans_encoder = self.init_trans_encoder(args)
        if args.model == 'AdaptiveFusion': 
            _model = Models.AdaptiveFusion(args, backbone, trans_encoder)
        elif args.model == 'DiffusionFusion':    
            diffusion_decoder = self.init_diffusion(args)
            # # load pretrained autoencoder 
            # vae = self.init_vae(args)
            # state_dict = torch.load('models/pretrained_vae.pth')
            # vae = load_and_freeze(vae, state_dict, freeze=args.freeze_backbone)
            # load pretrianed fusion model
            _model = Models.DiffusionFusion(args, backbone, trans_encoder, diffusion_decoder)
            if args.freeze_backbone: 
                state_dict = torch.load('models/pretrained_fusion_model.pth')
                _model = load_and_freeze(_model, state_dict, freeze=True, protect_list=['trans_encoder', 'upsampling'])            
        else: 
            raise NotImplementedError
        # update configs to enable attention outputs
        if args.show_att and args.model != 'AdaptiveDiffFusion':
            setattr(_model.trans_encoder[-1].config,'output_attentions', True)
            setattr(_model.trans_encoder[-1].config,'output_hidden_states', True)
            _model.trans_encoder[-1].bert.encoder.output_attentions = True
            _model.trans_encoder[-1].bert.encoder.output_hidden_states =  True
            for iter_layer in range(4):
                _model.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
            setattr(_model.trans_encoder[-1].config,'device', args.device)
        
        # IMPORTANT: FREEZE UNUSED LAYERS
        unused_keywords = ['image_backbone.classifier', 'bert.embeddings', 'bert.pooler', 'graph_conv.skip_conv']
        for n, param in _model.named_parameters(): 
            for keyword in unused_keywords: 
                if keyword in n: 
                    param.requires_grad = False
        
        return _model
      
    @staticmethod
    def init_backbone(args): 
        # init ImageNet pre-trained backbone model
        if args.arch=='hrnet':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            img_backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
        elif args.arch=='hrnet-w64':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            img_backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
        else:
            img_backbone = torchvision.models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            img_backbone = torch.nn.Sequential(*list(img_backbone.children())[:-2])
        if args.model == 'PointWImageFeat':
            mlps = [2048, 4096, 4096, 2048]
        elif args.use_point_feat:
            mlps = [3, 128, 128, 1024, 1024, 2048]
        else:
            mlps = [0, 128, 128, 1024, 1024, 2048]
        radar_backbone = PointnetSAModule(npoint=args.num_clusters, radius=0.4, nsample=32, mlp=mlps.copy())
        # depth_backbone = PointnetSAModule(npoint=49, radius=0.4, nsample=64, mlp=mlps.copy())
        backbone = dict(radar=radar_backbone, image=img_backbone)   #, depth=depth_backbone)
        return backbone
    
    @staticmethod
    def init_trans_encoder(args): 
        trans_encoder = []
        input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
        hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
        output_feat_dim = input_feat_dim[1:] + [args.output_feat_dim]  
        # max position embeddings according to mesh
        max_position_embeddings_dict = {'smpl': 455, 'smplh36m': 445, 'smplx': 677}
        max_position_embeddings = max_position_embeddings_dict[args.mesh_type]
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
            else:
                config.graph_conv = False
            config.mesh_type = args.mesh_type
            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
            for param in update_params:
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    setattr(config, param, arg_param)
            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config) 
            trans_encoder.append(model)
        trans_encoder = torch.nn.Sequential(*trans_encoder)
        return trans_encoder   
    
    @staticmethod
    def init_diffusion(args): 
        # DiT = DiT_models['DiT-S/8']
        # DiT = DiT_models['DiT-B/8']
        # dit = DiT(in_channels=args.vae_latent_dim, condition_channels=512)
        # ema = deepcopy(dit)
        # requires_grad(ema, False)
        diffusion = create_diffusion(timestep_respacing="", learn_sigma=False, predict_xstart=True, use_kl=False)  # default: 1000 steps, linear noise schedule
        return {
            'diffusion': diffusion, 
            # 'dit': dit, 
            # 'ema': ema
        }
    
    @staticmethod
    def init_vae(args): 
        vae = AutoEncoder(
            args, 
            in_channels=3, 
            latent_dim=args.vae_latent_dim, 
            hidden_feat_dim=args.vae_hidden_feat_dim
        )
        return vae
    
    def tic(self): 
        self.tic_time = time.time()
        
    def toc(self, label=''): 
        self.toc_time = time.time()
        if False: 
            self.print(label, 'cost time:', self.toc_time - self.tic_time)        
    
        
def vertices_loss(criterion_vertices, pred_vertices, gt_vertices, device):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    if len(gt_vertices) > 0:
        return criterion_vertices(pred_vertices, gt_vertices)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device) 

def mean_per_joint_position_error(pred, gt):
    """ 
    Compute mPJPE
    """
    with torch.no_grad():
        gt_pelvis = (gt[:, 2,:] + gt[:, 3,:]) / 2
        gt = gt - gt_pelvis[:, None, :]
        pred_pelvis = (pred[:, 2,:] + pred[:, 3,:]) / 2
        pred = pred - pred_pelvis[:, None, :]
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def mean_per_vertex_error(pred, gt):
    """
    Compute mPVE
    """
    with torch.no_grad():
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error
    
def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, device):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    conf = 1.
    if gt_keypoints_3d.shape[2] == 4:
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
    if len(gt_keypoints_3d) > 0:
        return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(device) 
    
def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence (conf) is binary and indicates whether the keypoints exist or not.
    """
    conf = 1.
    if gt_keypoints_2d.shape[2] == 3:
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_2d = gt_keypoints_2d[:, :, :-1]
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d)).mean()
    return loss

def visualize_mesh(renderer, images, gt_keypoints_2d, pred_vertices, pred_camera, pred_keypoints_2d):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(14))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # Do visualization for the first 6 images of the batch
    for i in range(min(batch_size, 10)):
        img = images[i].cpu().numpy().transpose(1,2,0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))   
    rend_imgs = torchvision.utils.make_grid(rend_imgs, nrow=1)
    return rend_imgs

def save_checkpoint(model, args, epoch, iteration, num_trial=10):
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint',
        'checkpoint-{}-{}'.format(epoch, iteration))
    # if not is_main_process():
    if dist.get_rank() != 0:
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, 'state_dict.bin'))
            torch.save(args, os.path.join(checkpoint_dir, 'training_args.bin'))
            torch.save(model_to_save, os.path.join(checkpoint_dir, 'model.bin'))
            print("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        print("Failed to save checkpoint after {} trails.".format(num_trial))
        
def load_and_freeze(model, state_dict, freeze=False, protect_list=None): 
    model.load_state_dict(state_dict, strict=False)
    # freeze the pretrained parameters
    if freeze: 
        for n, param in model.named_parameters(): 
            if n in state_dict: 
                param.requires_grad = False
            # enable specific models
            for i in protect_list: 
                if i in n: 
                    param.requires_grad = True
    return model    
            