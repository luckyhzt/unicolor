import os, sys
import os, sys
sys.path.append(os.path.abspath('.'))
sys.path.pop(0)

import torch
import argparse
import yaml
import numpy as np
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, random_split

from vqgan.models.hybrid_vqgan import VQModel
from vqgan.models.vqperceptual import VQLPIPSWithDiscriminator
from datasets.utils import get_dataloaders


def evaluate(args):
    # Load config
    img_dir = os.path.join(args.dir, f'images_{args.step}')
    os.makedirs(img_dir, exist_ok=True)
    config_path = os.path.join(args.dir, 'config.yaml')
    with open(config_path, 'rb') as fin:
        config = yaml.safe_load(fin)
    model_config = config['model']
    dataset_config = config['dataset']
    loss_config = config['loss']

    # Load dataset
    dataset_config['name'] = 'imagenet'
    dataset_config['num_workers'] = 0
    dataset_config['datapath'] = 'C:/MyFiles/Dataset/imagenet/full'
    dataset_config['color'] = args.color
    [valid_dl] = get_dataloaders(**dataset_config, splits=['val'])

    # Load pretrained model
    model = VQModel.load_from_checkpoint(
        args.checkpoint,
        ddconfig=model_config['ddconfig'],
        loss=VQLPIPSWithDiscriminator(**loss_config),
        n_embed=model_config['n_embed'],
        embed_dim=model_config['embed_dim'],
        learning_rate=0.0,
    )
    model.eval().cuda()

    # Evaluation
    valid_ds = valid_dl.dataset
    indices = np.random.randint(len(valid_ds), size=args.num_samples)
    for i in indices:
        x, _ = valid_ds[i]
        x = x.cuda()
        x = x.unsqueeze(0)
        # Gray input to color model
        if x.shape[1] == 1 and model_config['ddconfig']['in_channels'] == 3:
            x = x.repeat(1, 3, 1, 1)
        rec, _ = model(x)

        x = output_to_pil(x[0])
        rec = output_to_pil(rec[0])

        # Save result
        x.save(os.path.join(img_dir, f'{i}.jpg'))
        rec.save(os.path.join(img_dir, f'{i}_rec.jpg'))


def output_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
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

    parser.add_argument('--dir', type=str, default='vqgan/result/imagenet_hybrid')
    parser.add_argument('--step', type=str, default='309999')
    parser.add_argument('--color', type=str, default='color')
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()
    args.checkpoint = find(args.dir, args.step+'.ckpt')
    np.random.seed(100)
    evaluate(args)