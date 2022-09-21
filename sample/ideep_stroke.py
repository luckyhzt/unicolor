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
import random

from utils_func import *
from html_images import *



def put_point(input_ab,mask,loc,p,rgb):
    # input_ab    2x256x256    current user ab input (will be updated)
    # mask        1x256x256    binary mask of current user input (will be updated)
    # loc         2 tuple      (h,w) of where to put the user input
    # p           scalar       patch size
    # rgb         3 tuple      (r,g,b) value of user input
    rgb = np.array(rgb).astype(np.uint8)
    if len(rgb.shape) == 1:
        rgb = np.expand_dims(rgb, axis=[0, 1])
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    val = lab[:, :, 1:3].astype(np.int32) - 127
    val = np.transpose(val, (2, 0, 1))

    input_ab[:,loc[0]:loc[0]+p,loc[1]:loc[1]+p] = val
    mask[:,loc[0]:loc[0]+p,loc[1]:loc[1]+p] = 1
    return (input_ab,mask)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--cnn_dir', type=str, default='C:/MyFiles/ColorizationTran/other_colorizations/ideepcolor')

    args = parser.parse_args()

    # Load CNN
    os.chdir(args.cnn_dir)
    sys.path.append(args.cnn_dir)
    from data import colorize_image as CI
    colorModel = CI.ColorizeImageTorch(Xd=256)
    colorModel.prep_net(int(args.device[-1]), path='models/pytorch/caffemodel.pth')


    # Sampling arguments
    img_size = [256, 256]
    stroke_path = 'C:\\MyFiles\\CondTran\\data\\coco_strokes.json'
    dataset_path = 'C:\\MyFiles\\Dataset\\coco\\val2017'
    save_dir = 'C:\\Users\\lucky\\Desktop\\ideep_stroke_imagenet'
    num_strokes = [2, 16]

    html = HTML(save_dir, 'Sample')
    html.add_header('Ideep')
    
    # Load strokes
    with open(stroke_path, 'r') as file:
        all_strokes = json.load(file)
        print(len(all_strokes))

    pbar = tqdm(enumerate(all_strokes))


    random.seed(100)
    for i, file in pbar:
        filename = file['image']
        strokes = file['strokes']
        n_strokes = random.randint(num_strokes[0], num_strokes[1])
        n_strokes = min(n_strokes, len(strokes))
        strokes = random.sample(strokes, k=n_strokes)
        name = filename.split('.')[0]
        img_path = os.path.join(dataset_path, filename)

        gen_imgs = []

        I_color = Image.open(img_path).convert('RGB')
        I_gray = I_color.convert('L')

        gen_imgs.append(I_color)

        # Draw strokes
        draw_img = I_gray.copy().resize(img_size).convert('RGB')
        draw_img = draw_strokes(draw_img, img_size, strokes)

        gen_imgs.append(draw_img.resize(I_color.size))

        # CNN colorization
        input_ab = np.zeros((2,256,256))
        mask = np.zeros((1,256,256))
        for stk in strokes:
            (input_ab, mask) = put_point(input_ab, mask, stk['index'], 16, stk['color'])

        colorModel.load_image(img_path) # load an image
        img_out = colorModel.net_forward(input_ab, mask) # run model, returns 256x256 image
        mask_fullres = colorModel.get_img_mask_fullres() # get input mask in full res
        img_in_fullres = colorModel.get_input_img_fullres() # get input image in full res
        img_out_fullres = colorModel.get_img_fullres() # get image at full resolution

        gen_imgs.append(Image.fromarray(mask_fullres))
        gen_imgs.append(Image.fromarray(img_in_fullres))
        gen_imgs.append(Image.fromarray(img_out_fullres))
        
        save_result(html, index=i, images=gen_imgs)
        html.save()
