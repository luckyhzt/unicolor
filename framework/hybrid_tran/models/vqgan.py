import torch
from torch import nn
from hybrid_tran.utils.ops import *
from chroma_vqgan.models.vqgan import VQModel


class Chroma_VQGAN(nn.Module):
    def __init__(self, model_config, model_path=None):
        super().__init__()

        self.model = VQModel(
            ddconfig=model_config['ddconfig'],
            loss_config=None,
            n_embed=model_config['n_embed'],
            embed_dim=model_config['embed_dim'],
            learning_rate=0.0,
        )
        
        if model_path is not None:
            self.model.load_from_checkpoint(checkpoint_path=model_path, strict=True)

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

