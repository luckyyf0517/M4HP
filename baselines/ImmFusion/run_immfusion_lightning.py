import sys
sys.path.append('.')

import os
import yaml
import argparse

import torch
import torch.utils
import torchvision
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module='scipy')

from src.utils.miscellaneous import mkdir, set_seed
from src.model_interface import MInterface
from src.data_interface import DInterface


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Data related arguments
    parser.add_argument("--data_path", default='/home/nesc525/drivers/6/mmBody', type=str, required=False,
                        help="Directory with all datasets, each in one subfolder")
    parser.add_argument("--num_workers", default=4, type=int, 
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int, 
                        help="adjust image resolution.") 
    parser.add_argument("--test_scene", type=str, default='lab1')
    parser.add_argument("--seq_idxes", type=str, default='') 
    parser.add_argument('--skip_head', type=int, default=0)
    parser.add_argument('--skip_tail', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='mmBodyDataset')
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--add_rgb', action="store_true", help='add rgb values')
    parser.add_argument('--inputs', type=str, default='image0,radar0', help='input data')
    parser.add_argument('--trans_coor_to_cam', action="store_true")
    parser.add_argument("--mesh_type", default='smplx', type=str, help="smplx or smpl") 
    parser.add_argument('--mask_ratio', type=float, default=0.3)
    parser.add_argument('--num_clusters', type=int, default=49)
    parser.add_argument('--mask_limbs', action="store_true")
    parser.add_argument('--point_mask_ratio', type=float, default=0.99)
    parser.add_argument('--num_mask_limbs', type=int, default=None)
    parser.add_argument('--need_augm', action="store_true")
    parser.add_argument('--mix_data', action="store_true")
    parser.add_argument('--eval_test_dataset', action="store_true")
    
    # Loading/saving checkpoints
    parser.add_argument("--model_name_or_path", default='src/modeling/bert/bert-base-uncased', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--output_dir", default='output/output', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    
    # Training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=10, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=10, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float, 
                        help="The initial lr.")
    parser.add_argument("--num_train_epochs", default=50, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--vertices_loss_weight", default=100.0, type=float)
    parser.add_argument("--joints_loss_weight", default=1000.0, type=float)
    parser.add_argument("--score_loss_weight", default=100.0, type=float)
    parser.add_argument("--diffusion_loss_weight", default=1.0, type=float)
    parser.add_argument("--vloss_w_full", default=0.33, type=float) 
    parser.add_argument("--vloss_w_sub", default=0.33, type=float) 
    parser.add_argument("--vloss_w_sub2", default=0.33, type=float) 
    parser.add_argument("--drop_out", default=0.1, type=float, 
                        help="Drop out ratio in BERT.")
    
    # Model architectures
    parser.add_argument('-a', '--arch', default='hrnet-w64', 
                        help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--model", default='AdaptiveFusion', type=str,
                        help='Choose the model')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,64', type=str, 
                        help="The Image Feature Dimension.")   
    parser.add_argument('--output_feat_dim', default=3, type=int)
    parser.add_argument("--which_gcn", default='0,0,1', type=str, 
                        help="which encoder block to have graph conv. Encoder1, Encoder2, Encoder3. Default: only Encoder3 has graph conv") 
    parser.add_argument("--interm_size_scale", default=2, type=int)
    parser.add_argument('--use_point_feat', action="store_true", help='use point feature')
    
    parser.add_argument('--vae_latent_dim', default=4, type=int)
    parser.add_argument('--vae_hidden_feat_dim', default=768, type=int)
    
    # Others
    parser.add_argument('--train', dest="train", action="store_true", help='train or test')
    parser.add_argument('--pretrain_vae', dest="pretrain_vae", action="store_true")
    parser.add_argument('--visual', dest="visual", action="store_true", help='visual')
    parser.add_argument('--pause_at_start', action="store_true")
    parser.add_argument('--logging_steps', type=int, default=1000, 
                        help="Log every X steps.")
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument("--gpu_idx", type=str, default='[0]', help="select gpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    parser.add_argument("--local_rank", type=int, default=0, 
                        help="For distributed training.")
    parser.add_argument('--max_num_batch', type=int, default=10000)
    parser.add_argument('--save_snapshot', action="store_true")
    parser.add_argument('--points_w_image_feat', action="store_true")
    parser.add_argument('--fix_modalities', action="store_true")
    parser.add_argument('--wo_GIM', action="store_true")
    parser.add_argument('--wo_MMM', action="store_true")
    parser.add_argument('--wo_local_feat', action="store_true")
    parser.add_argument('--show_att', action="store_true")
    parser.add_argument('--joint_id', type=int, default=0)
    parser.add_argument('--calib_emb', action="store_true")
    parser.add_argument('--shuffle_inputs', action="store_true")
    parser.add_argument('--joints_2d_loss', type=str, default='')
    parser.add_argument('--freeze_backbone', action="store_true")
    parser.add_argument('--load_joints_2d', action="store_true")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    # args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    # os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    args.gpu_idx = eval(args.gpu_idx)
    args.num_gpus = len(args.gpu_idx)
    # Setup distributed 
    args.distributed = args.num_gpus > 1
    # torch.cuda.set_device(args.gpu_idx)
    args.device = torch.device(args.device)
    # Setup logging
    mkdir(args.output_dir)
    # Setup random seed
    set_seed(args.seed, args.num_gpus)
    # Setup input dict
    args.inputs = args.inputs.replace(' ', '').split(',')
    if args.model == 'TokenFusion':
        args.enabled_inputs = args.inputs
        args.inputs = ['image0', 'image1', 'depth0', 'depth1', 'radar0']
    args.input_dict = {}
    for m in ['image', 'depth', 'radar']:
        args.input_dict[m] = [i for i in args.inputs if m in i]
    if not args.inputs:
        raise RuntimeError("No input modality!")
    return args


if __name__ == '__main__': 
    args = parse_args()
    
    # load model 
    model = MInterface(args=args)
    data = DInterface(args=args)
    
    logger = TensorBoardLogger(save_dir='', name='', version=args.output_dir, sub_dir='tensorboard')
    
    checkpoint_callback = ModelCheckpoint(
        monitor='epoch',
        filename='{epoch:02d}',
        save_top_k=-1,
        mode='max',
        save_last=True,
        save_weights_only=False,
    )
    
    # Default used by the Trainer (no scaling of batch size)
    trainer = Trainer(
        default_root_dir=args.output_dir,
        strategy='ddp', #'ddp_find_unused_parameters_true',
        logger=logger,
        log_every_n_steps=1,
        devices=args.gpu_idx, 
        max_epochs=args.num_train_epochs, 
        reload_dataloaders_every_n_epochs=10,
        num_sanity_val_steps=1, # run validation step experimentaly
        callbacks=[checkpoint_callback], 
        enable_progress_bar=False, 
        use_distributed_sampler=True)

    if args.train:
        trainer.fit(model, datamodule=data, ckpt_path=args.resume_checkpoint if args.resume_checkpoint else None)
    else: 
        trainer.test(model, datamodule=data, ckpt_path=args.resume_checkpoint if args.resume_checkpoint else None)