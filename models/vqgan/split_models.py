import os, sys
sys.path.append(os.path.abspath('.'))
sys.path.pop(0)

import torch
import argparse
import yaml
import numpy as np
import pytorch_lightning as pl
from PIL import Image

from vqgan.models.hybrid_vqgan import VQModel
from vqgan.models.vqperceptual import VQLPIPSWithDiscriminator


def split(args):
    # Load config
    config_path = os.path.join(args.checkpoint, 'config.yaml')
    ckpt_path = os.path.join(args.checkpoint, 'last.ckpt')
    with open(config_path, 'rb') as fin:
        config = yaml.safe_load(fin)
    model_config = config['model']
    dataset_config = config['dataset']
    loss_config = config['loss']

    # Load pretrained model
    model = VQModel.load_from_checkpoint(
        ckpt_path,
        ddconfig=model_config['ddconfig'],
        loss=VQLPIPSWithDiscriminator(**loss_config),
        n_embed=model_config['n_embed'],
        embed_dim=model_config['embed_dim'],
        learning_rate=0.0,
    )
    save_path = os.path.join(args.checkpoint, 'models')
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.encoder, os.path.join(save_path, 'encoder.model'))
    torch.save(model.quant_conv, os.path.join(save_path, 'quant_conv.model'))
    torch.save(model.quantize, os.path.join(save_path, 'quantize.model'))
    torch.save(model.post_quant_conv, os.path.join(save_path, 'post_quant_conv.model'))
    torch.save(model.decoder, os.path.join(save_path, 'decoder.model'))
    torch.save(model.gray_encoder, os.path.join(save_path, 'gray_encoder.model'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='pretrained/imagenet_hybrid')
    args = parser.parse_args()
    split(args)