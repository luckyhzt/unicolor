import os, sys
from torch.utils import data

from torch.utils.data.dataloader import DataLoader

#os.system('pip install urllib3==1.21.1')
#os.system('pip install pytorch_lightning==1.3.7.post0')
#os.system('pip install nltk')

import argparse
import yaml
from datetime import timedelta
import torch

#from chroma_vqgan.models.vqgan import VQModel
#from chroma_vqgan.models.vqperceptual import VQLPIPSWithDiscriminator
from datasets.image_dataset import get_dataloaders

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers


def train(args):
    with open(args.config, 'rb') as fin:
        config = yaml.safe_load(fin)
    loss_config = config['loss']
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


    '''# Load config

    # Load dataset
    [train_dl, valid_dl] = get_dataloaders(**dataset_config, splits=['train', 'val'])

    # Build model
    loss_criterion = VQLPIPSWithDiscriminator(**loss_config)
    train_config['learning_rate'] = train_config['base_learning_rate'] * dataset_config['batch_size'] \
                                    * train_config['gpus'] * train_config['accumulate_grad_batches']
    model = VQModel(
        ddconfig=model_config['ddconfig'],
        loss=loss_criterion,
        n_embed=model_config['n_embed'],
        embed_dim=model_config['embed_dim'],
        learning_rate=train_config['learning_rate'],
        lr_decay=model_config['lr_decay'],
        )
    print(f"Setting learning rate to {train_config['learning_rate']}" )
    
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
    trainer.fit(model=model, train_dataloader=train_dl, val_dataloaders=valid_dl)'''



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='testing')

    parser.set_defaults(run=train)

    args, unknown = parser.parse_known_args()

    current_path = os.path.dirname(os.path.realpath(__file__))
    args.config = os.path.join(current_path, 'chroma_vqgan', 'configs', args.config + '.yaml')

    args.run(args)