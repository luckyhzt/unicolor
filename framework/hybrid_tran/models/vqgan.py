import os, sys
import torch
import argparse
import yaml
import numpy as np
import pytorch_lightning as pl
from PIL import Image
from torch import nn
from hybrid_tran.utils.ops import *
from chroma_vqgan.models.vqgan import VQModel


class Chroma_VQGAN(nn.Module):
    def __init__(self, model_path, load_vqgan_from_separate_file):
        super().__init__()

        with open(os.path.abspath( os.path.join(model_path, os.pardir, 'config.yaml') ), 'rb') as fin:
            config = yaml.safe_load(fin)
        
        model_config = config['model']
        if load_vqgan_from_separate_file:
            self.model = VQModel.load_from_checkpoint(checkpoint_path=model_path, strict=False,
                ddconfig=model_config['ddconfig'],
                loss_config=None,
                n_embed=model_config['n_embed'],
                embed_dim=model_config['embed_dim'],
                learning_rate=0.0,
            )
        else:
            # The model should be loaded along with the checkpoint of transformer
            self.model = VQModel(
                ddconfig=model_config['ddconfig'],
                loss_config=None,
                n_embed=model_config['n_embed'],
                embed_dim=model_config['embed_dim'],
                learning_rate=0.0,
            )
        self.encoder = self.model.encoder
        self.gray_encoder = self.model.gray_encoder
        self.quant_conv = self.model.quant_conv
        self.quantize = self.model.quantize
        self.post_quant_conv = self.model.post_quant_conv
        self.decoder = self.model.decoder
    
    def encode(self, x_color, x_gray):
        if x_gray != None:
            f_gray = self.gray_encoder(x_gray)
        else:
            f_gray = None

        if x_color != None:
            h = self.encoder(x_color)
            h = self.quant_conv(h)
            quant, emb_loss, info = self.quantize(h)
            color_idx = info[2].view(quant.shape[0], -1)
        else:
            color_idx = None
            
        return color_idx, f_gray
    
    def decode(self, idx_color, f_gray):
        q_shape = [f_gray.shape[0], f_gray.shape[2], f_gray.shape[3]]
        quant = self.quantize.get_codebook_entry(idx_color.view(-1), q_shape)
        feat = torch.cat([quant, f_gray], dim=1)
        feat = self.post_quant_conv(feat)
        rec = self.decoder(feat)
        return rec

