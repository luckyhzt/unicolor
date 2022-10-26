import os
from pickletools import optimize
from numpy.lib.twodim_base import mask_indices
import torch
from torch import nn
from torch.nn.modules import module
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import numpy as np
import math
from argparse import ArgumentParser
import torch.nn.functional as F
import itertools
import kornia

from hybrid_tran.models.transformer import HybridTran
from hybrid_tran.models.vqgan import Chroma_VQGAN
from hybrid_tran.utils.ops import *
from datasets.utils import *



class Colorization(pl.LightningModule):
    def __init__(
        self,
        vqgan_config,
        learning_rate,
        cond_ratio,
        hybridtran_config,
        lr_decay=[100, 1.0],
        vqgan_path=None,
    ):
        super().__init__()
        self.lr = learning_rate
        self.lr_decay_step, self.lr_decay = lr_decay
        # Prepare pretrained vqgan
        self.chroma_vqgan = Chroma_VQGAN(vqgan_config, vqgan_path).eval().requires_grad_(False)
        # Build transformer model
        self.hybrid_tran = HybridTran(**hybridtran_config)
        # Mask token
        self.mask_token = int(hybridtran_config['vocab_color'])
        self.cond_ratio = cond_ratio


    def random_select_cond(self, cond, indices, mask):
        B = mask.shape[0]

        selected_cond = []
        selected_indices = []
        for b in range(B):
            r = torch.rand(1)[0]
            if r <= self.cond_ratio:
                mask_indices = (mask[b] != 1).nonzero()
                num_mask = mask_indices.shape[0]
                num = min(cond.shape[1], num_mask//8)
                select_num = torch.randint(low=1, high=num+1, size=(1,))
                if select_num > 0:
                    selected = torch.randperm(num)[:select_num]
                    selected_cond.append(cond[b, selected, :])
                    selected_indices.append(indices[b, selected, :])
        
        if len(selected_cond) == 0:
            selected_cond = None
            selected_indices = None
        else:
            selected_cond = torch.cat(selected_cond, axis=0)
            selected_indices = torch.cat(selected_indices, axis=0)

        return selected_cond, selected_indices
        
    
    def forward(self, color, mask, cond):
        if cond.shape[1] > 0:
            cond_indices = (mask == -1).nonzero()
            cond_indices = cond_indices.reshape(cond.shape[0], cond.shape[1], -1)
            # Random select number of conds
            cond, cond_indices = self.random_select_cond(cond, cond_indices, mask)
        else:
            cond = None
            cond_indices = None

        gray = rgb_to_gray(color)
        idx_color, f_gray = self.chroma_vqgan.encode(color, gray)
        f_gray = f_gray.view(f_gray.shape[0], f_gray.shape[1], -1)
        f_gray = f_gray.permute(0, 2, 1).contiguous()
        target = idx_color.clone()
        # Replace masked positions with [MASK] tokens
        idx_color = idx_color.reshape(mask.shape)
        idx_color[mask == 0] = self.mask_token

        logits = self.hybrid_tran(idx_color, f_gray, cond, cond_indices)

        return logits, target

        

    def training_step(self, batch, batch_idx):

        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        x, mask, cond = batch

        logits, target = self(x, mask, cond)
        
        loss = self.masked_crossentropy(logits, target, mask)

        self.log('train/cross_entropy', loss, prog_bar=True, on_step=True)
        self.log('train/scheduler_step', scheduler._step_count)
        self.log('train/learning_rate', optimizer.param_groups[0]['lr'])
        return {'loss': loss}

    
    def validation_step(self, batch, batch_idx):
        x, mask, cond = batch

        logits, target = self(x, mask, cond)
        loss = self.masked_crossentropy(logits, target, mask)

        return {'loss': loss}


    def validation_epoch_end(self, outputs):
        ce_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val/cross_entropy', ce_loss, prog_bar=True)


    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out
    

    def masked_crossentropy(self, logits, target, mask=None):
        # Get masked elements
        if mask is not None:
            logits = logits.reshape(list(mask.shape)+[-1])
            target = target.reshape(mask.shape)
            logits = logits[mask != 1]
            target = target[mask != 1]
        else:
            logits = logits.reshape(-1, logits.shape[-1])
            target = target.reshape(-1)
        # Apply loss
        loss = F.cross_entropy(logits, target)
        return loss



    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        #all_modules = itertools.chain(self.hybrid_tran.named_modules(), self.discriminator.named_modules())
        #all_parameters = itertools.chain(self.hybrid_tran.named_parameters(), self.discriminator.named_parameters())

        for mn, m in self.hybrid_tran.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('cond_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.hybrid_tran.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, betas=(0.9, 0.95))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay)
        return [optimizer], [lr_scheduler]
