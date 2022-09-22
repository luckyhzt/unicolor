import argparse
from email.mime import image
import os
from skimage.measure import regionprops
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage import io
from skimage import color
import numpy as np
import argparse
import cv2
import sys
import random
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import glob



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='The path of the dataset.')
    parser.add_argument('--save_path', default='')
    args = parser.parse_args()
    if args.save_path == '':
        args.save_path = args.path + '_slic'
    
    image_exts = ('*.jpg', '*.png', '*.JPEG')

    image_paths = []
    for exts in image_exts:
        image_paths.extend( glob.glob(os.path.join(args.path, exts)) )

    os.makedirs(args.save_path, exist_ok=True)

    img_size = [int(256*1.15), int(256*1.15)]

    for i in tqdm(range(len(image_paths))):
        path = image_paths[i]
        img = Image.open(path).resize(img_size).convert('RGB')
        img = np.array(img)

        slic = cv2.ximgproc.createSuperpixelSLIC(img)
        slic.iterate(5)

        # retrieve the segmentation result
        labels = slic.getLabels()
        quantize_img = img.copy()
        low = np.min(labels)
        high = np.max(labels)
        for c in range(low, high+1):
            quantize_img[labels == c, :] = np.mean(img[labels == c, :], axis=0)

        result = Image.fromarray(quantize_img)

        save_name = os.path.relpath(path, args.path)
        save_name = save_name.split('.')[0] + '.png'
        save_name = os.path.join(args.save_path, save_name)
        os.makedirs(os.path.split(save_name)[0], exist_ok=True)
        result.save(save_name)