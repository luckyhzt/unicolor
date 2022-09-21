import os, sys
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
from datasets.utils import get_dataloaders

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers



import numpy as np
from PIL import Image


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
    [train_dl, valid_dl] = get_dataloaders(**dataset_config, splits=['train', 'val'])

    # Build model
    model_config['learning_rate'] = train_config['base_learning_rate'] * dataset_config['batch_size'] \
                                    * train_config['gpus'] * train_config['accumulate_grad_batches']
    model_config['vqgan_path'] = os.path.join(model_config['vqgan_path'])
    model = Colorization(**model_config)
    print(f"Setting learning rate to {model_config['learning_rate']}")
    
    # Trainer
    logger = pl_loggers.TensorBoardLogger(save_dir=config['log_dir'])
    if 'ckpt_steps' in train_config and train_config['ckpt_steps'] > 0:
        checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath=config['log_dir'],
            save_top_k=-1,
            every_n_train_steps=train_config['ckpt_steps'],
        )
    else:
        checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath=config['log_dir'],
            save_top_k=-1,
        )
    trainer = pl.Trainer(
        max_steps=train_config['steps'],
        gpus=train_config['gpus'],
        accumulate_grad_batches=train_config['accumulate_grad_batches'],
        precision=train_config['precision'],
        callbacks=[checkpoint],
        logger=logger,
        progress_bar_refresh_rate=train_config['log_steps'],
        accelerator='ddp' if train_config['gpus'] > 1 else None,
        plugins=pl.plugins.DDPPlugin(find_unused_parameters=True) if train_config['gpus'] > 1 else None,
        resume_from_checkpoint=train_config['from_checkpoint'],
    )

    # Start training
    trainer.fit(model=model, train_dataloader=train_dl, val_dataloaders=valid_dl)



if __name__ == '__main__':
    # Check images
    

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='testing')

    parser.set_defaults(run=train)

    args, unknown = parser.parse_known_args()

    current_path = os.path.dirname(os.path.realpath(__file__))
    args.config = os.path.join(current_path, 'filltran', 'configs', args.config + '.yaml')

    args.run(args)