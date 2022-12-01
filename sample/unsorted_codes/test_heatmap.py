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
from sample_func import *
from colorizer import *

ckpt_file = 'C:/MyFiles/CondTran/finals/bert_final/logs/bert/epoch=14-step=142124.ckpt'
device = 'cuda:0'
colorizer = Colorizer(ckpt_file, device, [256, 256], load_clip=True, load_warper=False)

# Sampling arguments
topk = 100
num_samples = [5, 5, 5]
img_size = [256, 256]
sample_size = [img_size[0]//16, img_size[1]//16]

path = 'C:\\MyFiles\\Comparison\\text\\final_text\\images\\0\\3699.png'
caption = 'red sign and blue car'

I_color = Image.open(path).convert('RGB')
I_gray = I_color.convert('L')

strokes, heatmaps = colorizer.get_strokes_from_clip(I_gray, caption)

if len(heatmaps) > 1:
    heat = []
    for h in heatmaps:
        heat.append(np.array(h))
    heat = np.concatenate(heat, axis=1)
    heat = Image.fromarray(heat)
else:
    heat = heatmaps[0]

heat.show()

#draw_img = draw_strokes(I_gray, img_size, strokes)

#x_gray = preprocess(I_gray, img_size).to(device)

#gen = colorizer.sample(x_gray, strokes=strokes, topk=topk)

