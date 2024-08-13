import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modeling.bert import BertConfig, Graphormer
from src.modeling.bert.transformer import FFN, Transformer


class AutoEncoder(nn.Module): 
    def __init__(self, args, in_channels=3, latent_dim=8, hidden_feat_dim=512): 
        super().__init__()
        
        config_class, model_class = BertConfig, Graphormer
        config = config_class.from_pretrained(args.config_name if args.config_name \
            else args.model_name_or_path + '-var')
        config.device = args.device
        config.output_attentions = False
        config.hidden_dropout_prob = args.drop_out
        config.img_feature_dim = in_channels
        config.output_feature_dim = latent_dim * 2  # [mean, var]
        args.hidden_size = hidden_feat_dim
        args.intermediate_size = int(args.hidden_size * args.interm_size_scale)
        config.max_position_embeddings = 677
        config.graph_conv = True
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
        # build transformer encoder
        self.encoder = model_class(config=config) 
        # build transformer decoder
        config.img_feature_dim = latent_dim 
        config.output_feature_dim = in_channels 
        self.decoder = model_class(config=config) 
        
    def encode(self, x): 
        latent_dist = self.encoder(x)   # [b, n, 2c]
        mu, log_var = latent_dist.chunk(2, dim=-1)
        return Dist(mu, log_var)
    
    def decode(self, z): 
        out = self.decoder(z)
        return out
    
    def forward(self, x): 
        # ground truth data (normalized)
        latent_dist = self.encode(x)
        z = latent_dist.sample()
        xout = self.decode(z)
        
        reconst_loss = F.mse_loss(xout, x, reduction='mean')
        kl_loss = latent_dist.kl_loss()    
        return reconst_loss, kl_loss
    
        
class Dist: 
    def __init__(self, mu, log_var): 
        self.mu = mu
        self.log_var = log_var
    
    def sample(self): 
        std = torch.exp(0.5 * self.log_var)
        eps = torch.rand_like(std)
        return self.mu + std * eps
    
    def kl_loss(self): 
        return -0.5 * torch.mean(1 + self.log_var - self.mu.pow(2) - self.log_var.exp())
    