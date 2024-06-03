import sys

import torch.utils
sys.path.append('.')
sys.path.append('./baselines/HuPR')

import warnings
warnings.filterwarnings("ignore")

import os
import yaml
import torch
import torchvision

from modelInterface.model_interface import MInterface
from dataInterface.data_interface import DInterface
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from HuPR.models import HuPRNet
from HuPR.datasets.dataset import HuPR3D_horivert, HuPR3D_raw
from HuPR.main import obj, parse_arg

torch.set_float32_matmul_precision('high')


if __name__ == "__main__":

    with open('./baselines/HuPR/config/mscsa_prgcn_demo.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = obj(cfg)
    args = parse_arg()
    
    model = MInterface(HuPRNet, args, cfg)
    data = DInterface(batch_size=cfg.TRAINING.batchSize * args.sampling_ratio, num_workers=cfg.SETUP.numWorkers, dataset=HuPR3D_raw, cfg=cfg, args=args)
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='epoch',
        filename='{epoch:02d}',
        save_top_k=1,
        mode='max',
        save_last=True,
        save_weights_only=False,
    )
    
    logger = TensorBoardLogger(save_dir='/root', name='log', version=args.version)
    checkpoint_path = os.path.join(cfg.DATASET.logDir, args.version, 'checkpoints/last.ckpt')

    # Default used by the Trainer (no scaling of batch size)
    trainer = Trainer(
        devices=args.gpuIDs,
        max_epochs=cfg.TRAINING.epochs,
        default_root_dir=args.version,
        strategy="ddp",
        logger=logger,
        log_every_n_steps=False,
        enable_progress_bar=False,
        reload_dataloaders_every_n_epochs=False,
        callbacks=[checkpoint_callback], 
        num_sanity_val_steps=0,
        # use_distributed_sampler=False,
        )

    if args.eval:
        data.setup(stage='test')
        trainer.test(model, datamodule=data, ckpt_path=checkpoint_path if os.path.exists(checkpoint_path) else None)
    else:
        data.setup(stage='fit')
        trainer.fit(model, datamodule=data, ckpt_path=checkpoint_path if os.path.exists(checkpoint_path) else None)
        
    
    
    
    
    