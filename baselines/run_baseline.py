import sys

import torch.utils
sys.path.append('.')
sys.path.append('./baselines/HuPR')

import warnings
warnings.filterwarnings("ignore")

import os
import yaml
import torch
import argparse
import torchvision
from easydict import EasyDict as edict

from dataInterface.data_interface import DInterface
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# from HuPR.datasets.dataset import HuPR3D_raw
from HuPR.main import obj, parse_arg

torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)


def parse_arg(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--version', type=str, default='test', metavar='B',
                        help='directory of saving/loading')
    parser.add_argument('--config', type=str, default='mscsa_prgcn_demo.yaml', metavar='B',
                        help='directory of visualization')
    parser.add_argument('--gpu', default=[0], type=eval, help='IDs of GPUs to use')                        
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('-sr', '--sampling_ratio', type=int, default=1, help='sampling ratio for training/test (default: 1)')
    return parser.parse_args()
    

if __name__ == "__main__":

    args = parse_arg()
    with open('config/' + args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = edict(cfg)
    
    from modelInterface import MInterfaceHuPR
    model = MInterfaceHuPR(args, cfg)
    data = DInterface(batch_size=cfg.TRAINING.batch_size, num_workers=cfg.SETUP.num_workers, cfg=cfg, args=args)
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='epoch',
        filename='{epoch:02d}',
        save_top_k=-1,
        mode='max',
        save_last=True,
        save_weights_only=False,
    )
    
    logger = TensorBoardLogger(save_dir='/root', name='log', version=args.version)
    checkpoint_path = os.path.join(cfg.DATASET.log_dir, args.version, 'checkpoints/last.ckpt')

    # Default used by the Trainer (no scaling of batch size)
    trainer = Trainer(
        devices=args.gpu if not args.eval else [0],
        max_epochs=cfg.TRAINING.epochs,
        default_root_dir=args.version,
        strategy="ddp",
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=False,
        reload_dataloaders_every_n_epochs=False,
        callbacks=[checkpoint_callback], 
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
    )

    if args.eval:
        data.setup(stage='test')
        trainer.test(model, datamodule=data, ckpt_path=checkpoint_path if os.path.exists(checkpoint_path) else None)
    else:
        data.setup(stage='fit')
        trainer.fit(model, datamodule=data, ckpt_path=checkpoint_path if os.path.exists(checkpoint_path) else None)
        
    
    
    
    
    