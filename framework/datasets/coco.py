import os, sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data as Data
from PIL import Image, ImageDraw
from tqdm import tqdm
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import nltk
import torch
import kornia

from datasets.mask import Mask_Generator



class Coco(Data.Dataset):
    def __init__(self, datapath, split, color, resolution, superpixel=None, patch_size=None, mode_prob=None, strokes=None):
        super().__init__()
        assert split in ('train', 'val')
        self.color = color
        self.resolution = resolution
        self.patch_size = patch_size
        self.strokes = strokes
        self.datapath = datapath
        self.split = split
        self.superpixel = superpixel
        self.loadInfo()
        self.sample_shape = [int(resolution[0]//patch_size[0]), int(resolution[1]//patch_size[1])]
        if mode_prob != None:
            self.mask_generator = Mask_Generator(self.sample_shape, mode_prob, strokes)
        else:
            self.mask_generator = None
    

    def loadInfo(self):
        with open(os.path.join(self.datapath, f'filtered_{self.split}.json'), 'r') as f:
            self.infos = json.load(f)


    def __getitem__(self, index):
        img_path = os.path.join(self.datapath, f'{self.split}2017', self.infos[index]['image'])
        image = Image.open(img_path).convert('RGB')
        if self.superpixel != None:
            sp_path = os.path.join(self.datapath, f'{self.split}2017_{self.superpixel}', self.infos[index]['image'].replace('.jpg', '.png'))
            image_sp = Image.open(sp_path).convert('RGB')
        else:
            image_sp = image.copy()

        if self.split == 'train':
            x, x_sp = self.train_transform(image, image_sp)
        else:
            x, x_sp = self.test_transform(image, image_sp)

        if self.mask_generator != None:
            mask = self.mask_generator()

            rows, cols = np.where(mask == -1)

            cond = torch.zeros(self.strokes, 3)
            for i in range(rows.shape[0]):
                r0 = rows[i] * self.patch_size[0]
                r1 = r0 + self.patch_size[0]
                c0 = cols[i] * self.patch_size[1]
                c1 = c0 + self.patch_size[1]
                patch = x_sp[:, r0:r1, c0:c1].clone()
                patch = patch.reshape(patch.shape[0], -1).permute(1, 0)
                unique, counts = torch.unique(patch, dim=0, return_counts=True)
                cond[i, :] = unique[torch.argmax(counts)]
        else:
            mask = 0.0
            cond = 0.0

        return x, mask, cond


    def train_transform(self, image, image_sp):
        flip = random.randint(0, 1)
        # Image
        image = self.resize(image)
        h0, w0, h, w = T.RandomCrop.get_params(image, output_size=self.resolution)
        image = TF.crop(image, h0, w0, h, w)
        x = TF.to_tensor(image)
        x = self.preprocess(x)
        # Super pixel
        image_sp = self.resize(image_sp)
        image_sp = TF.crop(image_sp, h0, w0, h, w)
        x_sp = TF.to_tensor(image_sp)
        x_sp = self.preprocess(x_sp)
        # Flip
        if flip == 1:
            x = TF.hflip(x)
            x_sp = TF.hflip(x_sp)
        return x, x_sp

    def test_transform(self, image, image_sp):
        # Image
        image = self.resize(image)
        image = TF.center_crop(image, self.resolution)
        x = TF.to_tensor(image)
        x = self.preprocess(x)
        # Super pixel
        image_sp = self.resize(image_sp)
        image_sp = TF.center_crop(image_sp, self.resolution)
        x_sp = TF.to_tensor(image_sp)
        x_sp = self.preprocess(x_sp)
        return x, x_sp

    
    def resize(self, image):
        image = image.resize([int(self.resolution[0] * 1.15), int(self.resolution[1] * 1.15)])
        
        return image

    def preprocess(self, x):
        x = x * 2.0 - 1.0
        return x


    def __len__(self):
        return len(self.infos)