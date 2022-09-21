import sys
import os
import cv2
import torch
import numpy as np
import argparse
import yaml
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageOps
import json
import nltk
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.widgets import Button
import kornia
import torchvision

from utils_func import *



@torch.no_grad()
def get_more_prior(filltran, img, strokes, sample_shape=[16, 16]):
    device = next(filltran.parameters()).device

    prior = get_prior(filltran, img, strokes)

    img_size = [sample_shape[0]*16, sample_shape[1]*16]

    x_gray = preprocess(img, img_size).to(device)
    _, f_gray = filltran.hybrid_vqgan.encode(None, x_gray)
    idx, gen = sample_fill(filltran, f_gray, 1, patch_size=[16, 16], prior=prior)

    prior = filltran.mask_token * torch.ones([1]+sample_shape).to(device).long()
    for stk in strokes:
        index = stk['index']
        r, c = index
        r, c = r//sample_shape[0], c//sample_shape[1]
        r0, c0, r1, c1 = get_input_range(sample_shape[0], sample_shape[1], r, c, sample_shape=[3, 3])
        prior[:, r0:r1, c0:c1] = idx[:, r0:r1, c0:c1]
    return gen, prior


@torch.no_grad()
def get_prior(filltran, img, strokes, sample_shape=[16, 16]):
    device = next(filltran.parameters()).device

    img_size = [sample_shape[0]*16, sample_shape[1]*16]
    img = img.resize(img_size[::-1]).convert('RGB')

    '''prior = filltran.mask_token * torch.ones(sample_shape).to(device).long()
    for stk in strokes:
        index = stk['index']
        r, c = index
        color = stk['color']
        #r0, c0, r1, c1 = get_input_range(256, 256, r, c, [64, 64])
        draw_img = draw_color(img, color, [None, None, None, None])
        x_draw = preprocess(draw_img, img_size).to(device)
        idx, _ = filltran.hybrid_vqgan.encode(x_draw, None)
        idx = idx.reshape(sample_shape[0], sample_shape[1])
        prior[r//16, c//16] = idx[r//16, c//16]'''
    
    '''prior = filltran.mask_token * torch.ones(sample_shape).to(device).long()
    for stk in strokes:
        index = stk['index']
        r, c = index
        color = stk['color']
        draw_img = draw_color(img, color, [None, None, None, None])
        draw_img.show()
        x_draw = preprocess(draw_img, img_size).to(device)
        x_gray = preprocess(img.convert('L'), img_size).to(device)
        idx, f_gray = filltran.hybrid_vqgan.encode(x_draw, x_gray)
        recon = filltran.hybrid_vqgan.decode(idx, f_gray)
        recon = output_to_pil(recon[0])
        recon.show()
        idx = idx.reshape(sample_shape[0], sample_shape[1])
        prior[r//16, c//16] = idx[r//16, c//16]'''

    '''draw_img = img.copy()
    for stk in strokes:
        index = stk['index']
        r, c = index
        color = stk['color']
        draw_img = draw_color(draw_img, color, [r, r+16, c, c+16])
    
    x_draw = preprocess(draw_img, img_size).to(device)
    idx, _ = filltran.hybrid_vqgan.encode(x_draw, None)
    idx = idx.reshape(sample_shape[0], sample_shape[1])

    prior = filltran.mask_token * torch.ones(sample_shape).to(device).long()
    for stk in strokes:
        index = stk['index']
        r, c = index
        prior[r//16, c//16] = idx[r//16, c//16]'''

    draw_img = img.copy()
    prior = filltran.mask_token * torch.ones(sample_shape).to(device).long()
    for stk in strokes:
        index = stk['index']
        r, c = index
        color = stk['color']
        this_draw = Image.fromarray( np.array(draw_img)[r:r+16, c:c+16, :] )
        this_draw = draw_full_color(this_draw, color, [None, None, None, None])
        x_draw = preprocess(this_draw, [16, 16]).to(device)
        output_to_pil(x_draw[0]).show()
        idx, _ = filltran.hybrid_vqgan.encode(x_draw, None)
        idx = idx.flatten(0)
        print(idx)
        prior[r//16, c//16] = idx

    prior = prior.unsqueeze(0)

    return prior


@torch.no_grad()
def get_prior_from_stroke(filltran, org, img, groups, sample_shape=[16, 16], patch_size=[64, 64], sample_patch=[2, 2], topk=100):
    device = next(filltran.parameters()).device
    img_size = [sample_shape[0]*16, sample_shape[1]*16]

    img = img.resize(img_size)

    indices = []
    for coords in groups:
        coords = np.array(coords)
        indices.append(coords)
    indices = np.concatenate(indices, axis=0)
    center = np.round( indices.mean(axis=0) )

    # Get local small patch
    r0, c0, r1, c1 = get_input_range(img_size[0], img_size[1], center[0], center[1], patch_size)
    r0 = r0 // 16 * 16;  c0 = c0//16 * 16;  r1 = r0 + patch_size[0];  c1 = c0 + patch_size[1]
    patch = Image.fromarray( np.array(img)[r0:r1, c0:c1, :] )
    patch_indices = indices - np.array([r0, c0])  # Shift to patch position
    patch_indices = np.multiply(patch_indices, np.divide(np.array(img_size), np.array(patch_size)))
    patch_indices = patch_indices.astype(int).tolist()
    patch_prior = get_prior(filltran, patch, patch_indices)
    # Sample the patch with stroke
    w, h = org.size
    w0 = int(c0 / img_size[1] * w);  w1 = int(c1 / img_size[1] * w)
    h0 = int(r0 / img_size[0] * h);  h1 = int(r1 / img_size[0] * h)
    patch_org = Image.fromarray(np.array(org)[h0:h1, w0:w1])
    x_gray = preprocess(patch_org, img_size).to(device)
    _, f_gray = filltran.hybrid_vqgan.encode(None, x_gray)
    _, sampled_patch = sample_fill(filltran, f_gray, topk, sample_patch, sample_patch, prior=patch_prior)
    # Get prior for whole image
    x_patch = preprocess(sampled_patch, patch_size).to(device)
    patch_idx, _ = filltran.hybrid_vqgan.encode(x_patch, None)
    patch_idx = patch_idx.reshape([patch_size[0]//16, patch_size[1]//16])
    prior = filltran.mask_token * torch.ones(sample_shape).to(device).long()
    prior[r0//16:r1//16, c0//16:c1//16] = patch_idx
    prior = prior.unsqueeze(0)

    patch.show()
    sampled_patch.show()

    return prior



@torch.no_grad()
def sample_fill(model, x_gray, topk, patch_size=[1, 1], mask_size=[1, 1], prior=None, retain_prior=False, message=None):
    _, f_gray = model.hybrid_vqgan.encode(None, x_gray)
    B = f_gray.shape[0]
    sample_shape = [16, 16]
    rows, cols = f_gray.shape[2:4]

    if prior == None:
        color_idx = model.mask_token * torch.ones([1, rows, cols]).to(f_gray.device).long()
    else:
        color_idx = prior.clone()

    rrange = get_sample_range(rows, patch_size[0])
    crange = get_sample_range(cols, patch_size[1])
    total = len(rrange) * len(crange)
    i = 0
    for r in rrange:
        for c in crange:
            if retain_prior and color_idx[:, r, c] != model.mask_token:
                continue
            # Index range for input
            r0, c0, r1, c1 = get_input_range(rows, cols, r, c, sample_shape)
            # Index range for mask
            rm0, cm0, rm1, cm1 = get_mask_range(rows, cols, r, c, mask_size, [r0, c0, r1, c1])
            # Index range for predicted patch
            rp0, cp0, rp1, cp1, pos = get_predict_range(rows, cols, r, c, patch_size, [r0, c0, r1, c1])
            # Input gray feature
            cond = f_gray[:, :, r0:r1, c0:c1]
            cond = cond.reshape(cond.shape[0], cond.shape[1], -1)
            cond = cond.permute(0, 2, 1).contiguous()
            # Input color indices
            idx = color_idx[:, r0:r1, c0:c1].clone()
            # Mask neighboring positions
            idx[:, rm0:rm1, cm0:cm1] = model.mask_token

            idx = idx.reshape(idx.shape[0], -1)

            logits = model.coltran(idx, cond)
            logits = logits[:, pos, :]
            logits = logits.reshape(-1, logits.shape[-1])
            logits = model.top_k_logits(logits, topk)
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            ix = ix.reshape(B, rp1-rp0, cp1-cp0)
            color_idx[:, rp0:rp1, cp0:cp1] = ix

            i += 1
            if message is not None:
                message.emit(int(100 * i / total))

    gen = model.hybrid_vqgan.decode(color_idx, f_gray)
    gen = output_to_pil(gen[0])
    
    return color_idx, gen


def get_sample_range(length, sample_size):
    num = int( np.ceil( float(length) / float(sample_size) ) )
    start = sample_size // 2
    stop = start + (num - 1) * sample_size
    steps = np.linspace(start=start, stop=stop, num=num).astype(int)
    return list(steps)

def get_input_range(rows, cols, r, c, sample_shape):
    # Index range for input
    c0 = c - sample_shape[1] // 2
    c1 = c0 + sample_shape[1]
    if c0 < 0:
        c0 = 0
        c1 = c0 + sample_shape[1]
    if c1 > cols:
        c1 = cols
        c0 = c1 - sample_shape[1]
    r0 = r - sample_shape[0] // 2
    r1 = r0 + sample_shape[0]
    if r0 < 0:
        r0 = 0
        r1 = r0 + sample_shape[0]
    if r1 > rows:
        r1 = rows
        r0 = r1 - sample_shape[0]
    
    return int(r0), int(c0), int(r1), int(c1)

def get_mask_range(rows, cols, r, c, mask_size, input_range):
    # Index range for mask
    r0 = r - mask_size[0] // 2;  r1 = r0 + mask_size[0]
    c0 = c - mask_size[1] // 2;  c1 = c0 + mask_size[1]
    r0 = max(0, r0);  r1 = min(rows, r1)
    c0 = max(0, c0);  c1 = min(cols, c1)
    r_start = input_range[0];  c_start = input_range[1]
    return int(r0 - r_start), int(c0 - c_start), int(r1 - r_start), int(c1 - c_start)

def get_predict_range(rows, cols, r, c, patch_size, input_range):
    # Index range for predict patch
    r0 = r - patch_size[0] // 2;  r1 = r0 + patch_size[0]
    c0 = c - patch_size[1] // 2;  c1 = c0 + patch_size[1]
    r0 = max(0, r0);  r1 = min(rows, r1)
    c0 = max(0, c0);  c1 = min(cols, c1)

    pos = []
    for row in range(r0, r1):
        for col in range(c0, c1):
            nrow = row - input_range[0]
            ncol = col - input_range[1]
            width = input_range[3] - input_range[1]
            pos.append( int(nrow * width + ncol) )
    
    return int(r0), int(c0), int(r1), int(c1), pos


def color_resize(l, color):
    color = color.resize(l.size)
    resized = draw_color(l.convert('RGB'), color, [None, None, None, None])
    return resized

@torch.no_grad()
def upsample(model, l_org, color, patch_size, mask_size, topk, message=None):
    img_size = [(l_org.size[0] // 16) * 16, (l_org.size[1] // 16) * 16]
    l = l_org.resize(img_size)
    color = color.resize(img_size)

    x_gray = preprocess(l, img_size[::-1]).to(model.device)
    x_color = preprocess(color, img_size[::-1]).to(model.device)
    color_idx, f_gray = model.hybrid_vqgan.encode(x_color, x_gray)
    color_idx = color_idx.reshape(1, f_gray.shape[2], f_gray.shape[3])
    # Sample
    _, upsampled = sample_fill(model, f_gray, topk, patch_size, mask_size, prior=color_idx, message=message)
    # Append with original gray
    upsampled = draw_color(l.convert('RGB'), upsampled, [None, None, None, None])

    return upsampled
    