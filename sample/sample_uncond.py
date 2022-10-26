import sys
import os
import torch
import numpy as np
import yaml
from tqdm import tqdm
from PIL import Image
import glob

from utils_func import *
from sample_func import *
from colorizer import *


if __name__ == '__main__':
    ckpt_file = '/home/huangzhitong/code/unicolor/framework/logs/bert_coco/epoch=144-step=259999.ckpt'
    device = 'cuda:1'
    colorizer = Colorizer(ckpt_file, device, [256, 256], load_clip=False, load_warper=False)

    org_path = '/home/huangzhitong/code/unicolor/results/old_coco_259999/original'
    col_path = '/home/huangzhitong/code/unicolor/results/old_coco_259999/colorized'

    os.makedirs(org_path, exist_ok=True)
    os.makedirs(col_path, exist_ok=True)

    img_paths = glob.glob('/home/huangzhitong/dataset/coco/val2017/*.jpg')

    for path in tqdm(img_paths):
        img = Image.open(path).convert('RGB')
        gray = img.convert('L')
        colorized = colorizer.sample(gray, [], topk=100)
        name = path.split('/')[-1].split('.')[0]
        img.save(os.path.join(org_path, name+'.png'))
        colorized.save(os.path.join(col_path, name+'.png'))