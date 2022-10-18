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

from hybrid_tran.models.colorization import Colorization
from datasets.image_dataset import get_dataloaders

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
    dataset_config = config['data']

    # Save configs
    os.makedirs(config['log_dir'], exist_ok=True)
    with open(os.path.join(config['log_dir'], 'config.yaml'), 'w') as fout:
        yaml.dump(config, fout)

    # Load dataset
    train_dl = get_dataloaders(**dataset_config['train'])
    valid_dl = get_dataloaders(**dataset_config['val'])

    # Build model
    if 'learning_rate' not in train_config:
        train_config['learning_rate'] = train_config['base_learning_rate'] * dataset_config['train']['batch_size'] \
                                        * len(train_config['gpus']) * train_config['accumulate_grad_batches']
    model = Colorization(learning_rate=train_config['learning_rate'], **model_config)
    print(f"Setting learning rate to {train_config['learning_rate']}")
    
    # Trainer
    logger = pl_loggers.TensorBoardLogger(save_dir=config['log_dir'])
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=config['log_dir'],
        save_top_k=-1,
        every_n_train_steps=train_config['ckpt_steps'],
    )

    trainer = pl.Trainer(
        max_steps=train_config['steps'],
        gpus=train_config['gpus'],
        accumulate_grad_batches=train_config['accumulate_grad_batches'],
        precision=train_config['precision'],
        callbacks=[checkpoint],
        logger=logger,
        progress_bar_refresh_rate=train_config['log_steps'],
        accelerator='ddp' if len(train_config['gpus']) > 1 else None,
        plugins=pl.plugins.DDPPlugin(find_unused_parameters=True) if len(train_config['gpus']) > 1 else None,
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
    args.config = os.path.join(current_path, 'hybrid_tran', 'configs', args.config + '.yaml')

    args.run(args)