import os
import torch
from copy import deepcopy

def load_vae(): 
    model_dict = torch.load('/remote-home/iot_yanyifan/ImmFusion/output/autoencoder/checkpoints/last.ckpt', map_location='cpu')
    model_dict = model_dict['state_dict']
    model_dict_new = {}
    for k, v in model_dict.items(): 
        if 'model.' in k: 
            model_dict_new[k.replace('model.', '')] = v
    torch.save(model_dict_new, 'models/pretrained_vae.pth')


def remove_depth_backbone():
    model_dict = torch.load('/remote-home/iot_yanyifan/ImmFusion/output/fusediffusion-v5/checkpoints/epoch=14.ckpt', map_location='cpu')
    model_dict_new = deepcopy(model_dict)
    for k, v in model_dict['state_dict'].items(): 
        if 'model.depth_backbone' in k: 
            del model_dict_new['state_dict'][k]
    torch.save(model_dict_new, '/remote-home/iot_yanyifan/ImmFusion/output/fusediffusion-v5/checkpoints/epoch=14.ckpt')


if __name__ == '__main__': 
    remove_depth_backbone()
    print('done')
    
    