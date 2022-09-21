import os, sys
from matplotlib.pyplot import draw
from torch.utils import data

from torch.utils.data.dataloader import DataLoader

#os.system('pip install urllib3==1.21.1')
#os.system('pip install pytorch_lightning==1.3.7.post0')
#os.system('pip install nltk')
#os.system('pip install kornia==0.5.11')

import argparse
import yaml
from datetime import timedelta
import torch

from filltran.models.colorization import Colorization
from datasets.utils import get_dataloaders, output_to_pil

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import numpy as np
from PIL import Image
import cv2


def train(args):
    # Load config
    with open(args.config, 'rb') as fin:
        config = yaml.safe_load(fin)
    model_config = config['model']
    train_config = config['train']
    dataset_config = config['dataset']

    # Save configs
    os.makedirs(config['log_dir'], exist_ok=True)
    with open(os.path.join(config['log_dir'], 'config.yaml'), 'w') as fout:
        yaml.dump(config, fout)

    # Load dataset
    [valid_dl] = get_dataloaders(**dataset_config, splits=['val'])

    diter = iter(valid_dl)

    x, x_sp, mask, cond = diter.next()
    x, x_sp, mask, cond = diter.next()
    x, x_sp, mask, cond = diter.next()
    x, x_sp, mask, cond = diter.next()
    x, x_sp, mask, cond = diter.next()

    cond_indices = (mask == -1).nonzero()
    cond_indices = cond_indices.reshape(cond.shape[0], cond.shape[1], -1)

    draw_img = output_to_pil(x_sp[0])
    for i, ind in enumerate(cond_indices[0]):
        b, r, c = ind
        color = cond[0, i, :].numpy()
        color = (color + 1.0) / 2.0 * 255.0
        color = color.astype(np.uint8)
        color = np.expand_dims(color, axis=(0, 1))
        color = cv2.resize(color, (16-4, 16-4), interpolation=cv2.INTER_NEAREST)
        color = cv2.copyMakeBorder(color, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        draw_img = np.array(draw_img)
        draw_img[r*16:r*16+16, c*16:c*16+16, :] = color
        draw_img = Image.fromarray(draw_img)

    output_to_pil(x[0]).show()
    draw_img.show()
    
    '''draw_img = output_to_pil(x[0])
    for i, ind in enumerate(cond_indices[0]):
        b, r, c = ind
        color = cond_mean[0, i, :].numpy()
        color = (color + 1.0) / 2.0 * 255.0
        color = color.astype(np.uint8)
        color = np.expand_dims(color, axis=(0, 1))
        color = cv2.resize(color, (16-4, 16-4), interpolation=cv2.INTER_NEAREST)
        color = cv2.copyMakeBorder(color, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        draw_img = np.array(draw_img)
        draw_img[r*16:r*16+16, c*16:c*16+16, :] = color
        draw_img = Image.fromarray(draw_img)

    draw_img.show()'''






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='testing')

    parser.set_defaults(run=train)

    args, unknown = parser.parse_known_args()

    current_path = os.path.dirname(os.path.realpath(__file__))
    args.config = os.path.join(current_path, 'filltran', 'configs', args.config + '.yaml')

    args.run(args)