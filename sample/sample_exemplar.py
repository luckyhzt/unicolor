import sys
import os
import cv2
import importlib
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
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import clip
import kornia
import torchvision
import skimage.color

from utils_func import *
from html_images import *
from sample_func import *
from ImageMatch.warp import ImageWarper



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--root_dir', type=str, default='C:/MyFiles/CondTran/frameworks/bert_strokes')
    parser.add_argument('--log_dir', type=str, default='logs/bert_slic')
    parser.add_argument('--step', type=str, default='149999')

    args = parser.parse_args()

    # Load transformer
    os.chdir(args.root_dir)
    sys.path.append(args.root_dir)
    module = importlib.import_module(f'filltran.models.colorization')
    args.fill_model = getattr(module, 'Colorization')
    filltran = load_model(args.fill_model, args.log_dir, args.step).to(args.device).eval().requires_grad_(False)

    # Sampling arguments
    topk = 100
    num_samples = 1
    img_size = [256, 256]
    sample_size = [img_size[0]//16, img_size[1]//16]
    pair_path = 'C:/MyFiles/CondTran/data/all_exemplars.json'
    dataset_path = 'C:/MyFiles/Dataset/imagenet/val5000/val'
    ref_path = 'C:/MyFiles/Dataset/imagenet/full/train'
    save_dir = 'C:/MyFiles/CondTran/sample_result/slic_exemplar_2'

    html = HTML(save_dir, 'Sample')
    html.add_header(os.path.join(args.root_dir, args.log_dir, args.step))

    # Load pairs
    with open(pair_path, 'r') as file:
        all_pairs = json.load(file)
    #np.random.seed(10)
    np.random.shuffle(all_pairs)
    all_pairs = all_pairs[:200]

    # Load image warper
    warper = ImageWarper('cuda')

    i = 0
    pbar = tqdm(all_pairs)
    for file in pbar:
        filename = file['image']
        refname = file['exemplar']
        name = filename.split('.')[0]
        gen_imgs = []

        in_dir = os.path.join(dataset_path, filename)
        ref_dir = os.path.join(ref_path, refname)
        I_color = Image.open(in_dir).convert('RGB')
        I_gray = I_color.convert('L')
        I_ref = Image.open(ref_dir).convert('RGB')

        gen_imgs.append(I_color)
        gen_imgs.append(I_ref)

        warped_img, similarity_map = warper.warp_image(I_color, I_ref)
        gen_imgs.append(warped_img.resize(I_color.size))
        warped_img = warped_img.resize(img_size)

        similarity_map = cv2.resize(similarity_map, tuple(sample_size))
        similarity_map = similarity_map.reshape(-1)
        threshold = min(0.23, np.sort(similarity_map)[-10])
        indices = np.where( (similarity_map >= threshold))

        strokes = []
        warped_img = np.array(warped_img)
        for ind in indices[0]:
            index = [ind//16 * 16, ind%16 * 16]
            color = warped_img[index[0]:index[0]+16, index[1]:index[1]+16, :]
            color = color.mean(axis=(0, 1))
            strokes.append({'index': index, 'color': color.tolist()})

        draw_img = draw_strokes(I_gray.convert('RGB'), img_size, strokes)

        gen_imgs.append(draw_img.resize(I_color.size))

        x_gray = preprocess(I_gray, img_size).to(args.device)

        for n in range(num_samples):
            gen = filltran.sample(x_gray, topk, strokes)
            gen = output_to_pil(gen[0])
            gen_resize = color_resize(I_gray, gen)
            gen_imgs.append(gen_resize)

        save_result(html, index=i, images=gen_imgs)
        html.save()
        i += 1



    
