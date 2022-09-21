import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.pop(0)

import importlib
import torch
import numpy as np
import argparse
import logging
import yaml
import pickle
from tqdm import tqdm
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torchvision.transforms as T
import torch.nn.functional as F

from datasets.utils import *


def sample(args):
    # Load config
    config_path = os.path.join(args.dir, 'config.yaml')
    with open(config_path, 'rb') as fin:
        config = yaml.safe_load(fin)
    model_config = config['model']

    # Load model
    model_config['data_url'] = 'pretrained'
    model = args.model.load_from_checkpoint(
        args.checkpoint,
        **model_config,
    )
    model.eval().to(args.device)

    # Input image
    x = Image.open(args.img).convert('L')
    size = list(x.size)
    if x.size[0] > 800 or x.size[1] > 800:
        if x.size[0] > x.size[1]:
            size[0] = 800
            size[1] = int( 800.0 * x.size[1] / x.size[0] )
        else:
            size[1] = 800
            size[0] = int( 800.0 * x.size[0] / x.size[1] )
    
    img_shape = [(size[1] // 16) * 16, (size[0] // 16) * 16]
    transform = T.Compose([T.Resize(img_shape), T.ToTensor()])
    x = transform(x)
    x = x * 2 - 1
    x = x.unsqueeze(0).to(args.device)

    img_dir = args.img.replace('.', '_')
    os.makedirs(img_dir, exist_ok=True)

    ori = output_to_pil(x[0])

    ori.save(os.path.join(img_dir, f'ori.jpg'))

    sample_shape = [16, 16]
    with torch.no_grad():
        f_gray = model.hybrid_vqgan.gray_encoder(x)
        rows, cols = f_gray.shape[2:]

        for i in tqdm(range(args.num_samples)):
            color_idx = 4096 * torch.ones([1, rows, cols]).to(args.device).long()
            for r in range(rows):
                for c in range(cols):
                    c0 = c - sample_shape[1] // 2
                    c1 = c0 + sample_shape[1]
                    if c0 < 0:
                        c0 = 0
                        c1 = c0 + sample_shape[1]
                    if c1 > cols:
                        c1 = cols
                        c0 = c1 - sample_shape[1]
                    #r0 = max(r - sample_shape[0] + 1, 0)
                    #r1 = r0 + sample_shape[0]
                    r0 = r - sample_shape[0] // 2
                    r1 = r0 + sample_shape[0]
                    if r0 < 0:
                        r0 = 0
                        r1 = r0 + sample_shape[0]
                    if r1 > rows:
                        r1 = rows
                        r0 = r1 - sample_shape[0]
                    pos = (r - r0) * sample_shape[1] + (c - c0)

                    cond = f_gray[:, :, r0:r1, c0:c1]
                    cond = cond.reshape(cond.shape[0], cond.shape[1], -1)
                    cond = cond.permute(0, 2, 1).contiguous()
                    idx = color_idx[:, r0:r1, c0:c1]
                    idx = idx.reshape(idx.shape[0], -1)

                    logits = model.coltran(idx, cond)
                    logits = logits[:, pos, :]
                    logits = model.top_k_logits(logits, args.topk)
                    probs = F.softmax(logits, dim=-1)
                    ix = torch.multinomial(probs, num_samples=1)
                    color_idx[:, r, c] = ix.squeeze(1)

            gen = model.hybrid_vqgan.decode(color_idx, f_gray)
            gen = output_to_pil(gen[0])
            gen.save(os.path.join(img_dir, f'gen_{i}.jpg'))


def output_to_pil(x):
    x = x.detach()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    if x.is_cuda:
        x = x.cpu()
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    if x.shape[2] == 1:
        x = x[:, :, 0]
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def find(path, name):
    for root, dirs, files in os.walk(path):
        for f in files:
            if name in f:
                return os.path.join(root, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    
    parser.add_argument('--img', type=str, default='old_photos/17.jpg')
    parser.add_argument('--dir', type=str, default='logs/filltran_patch')
    parser.add_argument('--step', type=str, default='400379')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--topk', type=int, default=400)
    args = parser.parse_args()

    module = importlib.import_module(f'filltran.models.colorization')
    args.model = getattr(module, 'Colorization')
    args.checkpoint = find(args.dir, args.step+'.ckpt')

    sample(args)