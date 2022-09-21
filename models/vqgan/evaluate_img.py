import os, sys
import os, sys
# Change to current working directory
folder = os.path.join(os.path.abspath('.'), 'frameworks')
os.chdir(folder)
sys.path.append(folder)

import torch
import argparse
import yaml
import numpy as np
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T

from vqgan.models.hybrid_vqgan import VQModel
from vqgan.models.vqperceptual import VQLPIPSWithDiscriminator
from datasets.utils import get_dataloaders


def evaluate(args):
    # Load config
    #img_dir = os.path.join(args.image.replace('.', '_'))
    #os.makedirs(img_dir, exist_ok=True)
    config_path = os.path.join(args.dir, 'config.yaml')
    with open(config_path, 'rb') as fin:
        config = yaml.safe_load(fin)
    model_config = config['model']
    dataset_config = config['dataset']
    loss_config = config['loss']

    # Load dataset
    #dataset_config['name'] = 'imagenet'
    #dataset_config['num_workers'] = 0
    #dataset_config['datapath'] = 'C:/MyFiles/Dataset/imagenet/full'
    #dataset_config['color'] = args.color
    #[valid_dl] = get_dataloaders(**dataset_config, splits=['val'])

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
    #valid_ds = valid_dl.dataset
    #indices = np.random.randint(len(valid_ds), size=args.num_samples)
    #for i in indices:
    x = Image.open(args.image).convert('RGB')
    transform = T.Compose([T.Resize([(x.size[1]//16 * 16), (x.size[0]//16 * 16)]), T.ToTensor()])
    x = transform(x)
    x = x * 2 - 1
    x = x.unsqueeze(0).cuda()
    # Gray input to color model
    if x.shape[1] == 1 and model_config['ddconfig']['in_channels'] == 3:
        x = x.repeat(1, 3, 1, 1)
    
    with torch.no_grad():
        rec, _ = model(x)

    loss = (rec - x)**2
    loss = torch.sqrt(loss.mean())

    ori = output_to_pil(x[0])
    rec = output_to_pil(rec[0])

    # Show result
    ori.show()
    rec.show()

    with torch.no_grad():
        xrec, qloss = model(x)
        aeloss, log_dict_ae = model.loss(qloss, x, xrec, 0, model.global_step,
                                                last_layer=model.get_last_layer(), split="val")
        print(model.loss.discriminator(xrec))
        print(model.loss.discriminator(x))
    
    print(loss)


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

    parser.add_argument('--image', type=str, default='../data/raw/in6.jpg')
    parser.add_argument('--dir', type=str, default='logs/vqgan_imagenet_full')
    parser.add_argument('--step', type=str, default='279999')
    args = parser.parse_args()
    args.checkpoint = find(args.dir, args.step+'.ckpt')
    np.random.seed(100)
    evaluate(args)