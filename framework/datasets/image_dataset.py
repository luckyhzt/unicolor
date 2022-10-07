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
import torch

from mask import Mask_Generator



class Image_Dataset(Data.Dataset):
    def __init__(self, datapath, meta_file, resolution, split, patch_size=None, mode_prob=None, strokes=None):
        assert split in ('train', 'val')
        super().__init__()
        self.split = split
        self.meta_file = meta_file
        self.resolution = resolution
        self.patch_size = patch_size
        self.strokes = strokes
        self.datapath = datapath
        self.loadInfo()
        if mode_prob is not None:
            assert self.patch_size is not None and self.strokes is not None
            self.mask_generator = Mask_Generator([int(resolution[0]//patch_size[0]), int(resolution[1]//patch_size[1])], mode_prob, strokes)
        else:
            self.mask_generator = None
    

    def loadInfo(self):
        assert isinstance(self.meta_file, list) or isinstance(self.meta_file, tuple)
        self.metas = []
        for mfile in self.meta_file:
            with open(os.path.join(self.datapath, mfile), 'r') as f:
                self.metas += json.load(f)
        


    def __getitem__(self, index):
        img_meta = self.metas[index]
        img_path = os.path.join(self.datapath, img_meta['image_path'])
        image = Image.open(img_path).convert('RGB')

        if self.mask_generator is not None:
            sp_path = os.path.join(self.datapath, img_meta['slic_path'])
            image_sp = Image.open(sp_path).convert('RGB')
            if self.split == 'train':
                x, x_sp = self.train_transform(image, image_sp)
            else:
                x, x_sp = self.test_transform(image, image_sp)
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
            return x, mask, cond
        else:
            if self.split == 'train':
                x = self.train_transform(image)
            else:
                x = self.test_transform(image)
            return x


    def train_transform(self, image, image_sp=None):
        flip = random.randint(0, 1)
        # Image
        image = self.resize(image)
        h0, w0, h, w = T.RandomCrop.get_params(image, output_size=self.resolution)
        image = TF.crop(image, h0, w0, h, w)
        x = TF.to_tensor(image)
        x = self.preprocess(x)
        if flip == 1:
            x = TF.hflip(x)
        # Super pixel
        if image_sp is not None:
            image_sp = self.resize(image_sp)
            image_sp = TF.crop(image_sp, h0, w0, h, w)
            x_sp = TF.to_tensor(image_sp)
            x_sp = self.preprocess(x_sp)
            if flip == 1:
                x_sp = TF.hflip(x_sp)
            return x, x_sp
        else:
            return x

    def test_transform(self, image, image_sp=None):
        # Image
        image = self.resize(image)
        image = TF.center_crop(image, self.resolution)
        x = TF.to_tensor(image)
        x = self.preprocess(x)
        # Super pixel
        if image_sp is not None:
            image_sp = self.resize(image_sp)
            image_sp = TF.center_crop(image_sp, self.resolution)
            x_sp = TF.to_tensor(image_sp)
            x_sp = self.preprocess(x_sp)
            return x, x_sp
        else:
            return x
  
    def resize(self, image):
        image = image.resize([int(self.resolution[0] * 1.15), int(self.resolution[1] * 1.15)])
        
        return image

    def preprocess(self, x):
        x = x * 2.0 - 1.0
        return x


    def __len__(self):
        return len(self.metas)



# Test the dataset class
if __name__ == '__main__':
    ds = Image_Dataset(
        datapath='/home/huangzhitong/dataset/coco',
        meta_file=['train2017_meta.json', 'unlabeled2017_meta.json'],
        split='train'
        resolution=[256, 256],
    )

    

    print(len(ds))