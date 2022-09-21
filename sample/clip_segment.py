import sys
import os
import cv2
import torch
import numpy as np
import yaml
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageOps
import json
import nltk
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import clip
import kornia
import torchvision
import io
from scipy.ndimage.filters import gaussian_filter

from utils_func import *


def parse_caption(caption, colors):
    words = nltk.tokenize.word_tokenize(caption)
    # replace grey with gray
    words = replace(words, 'grey', 'gray')
    objects = {}
    for col in colors:
        if col in words:
            pos = words.index(col)
            objects[words[pos+1]] = col
    return objects


def c_mean_shift(image):
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.pyrMeanShiftFiltering(img, 16, 48)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.astype(np.uint8)

def split_gray_img(image, n_labels=2):
    img = image.copy()
    step = 255 // n_labels
    t = list(np.arange(0, 255, step)) + [255]
    for i, (t1, t2) in enumerate(zip(t[:-1], t[1:])):
        img[(img >= t1) & (img < t2)] = t1
    return img


@torch.no_grad()
def search_objects(model, device, preprocess, img, objects, sample_size=[16, 16], step_size=2.5, topk=2):
    img_size = list(img.size).copy()[::-1]
    patch_size = [(img_size[0] // sample_size[0]), (img_size[1] // sample_size[1])]
    img_size = [patch_size[0]*sample_size[0], patch_size[1]*sample_size[1]]
    img = img.resize(img_size[::-1])

    pad_size = np.max([patch_size[0] * step_size, patch_size[1] * step_size])
    pad_img = ImageOps.expand(img, border=int(pad_size), fill=0)

    times = np.zeros([len(objects), sample_size[0], sample_size[1]]) + 1e-6
    scores = np.zeros([len(objects), sample_size[0], sample_size[1]])
    with torch.no_grad():
        # Prepare text features
        text = clip.tokenize(objects).to(device)
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp()
        # Start searching
        for r in range(1, sample_size[0]-1):
            for c in range(1, sample_size[1]-1):
                img_array = np.array(pad_img)

                r0 = max(0, r - int(step_size//1.2));  r1 = min(sample_size[0], r + int(step_size//1.2) + 1)
                c0 = max(0, c - int(step_size//1.2));  c1 = min(sample_size[1], c + int(step_size//1.2) + 1)

                y = (r + 0.5) * patch_size[0] + pad_size
                x = (c + 0.5) * patch_size[1] + pad_size
                y0 = y - (step_size + 0.5) * patch_size[0];  y1 = y + (step_size + 0.5) * patch_size[0]
                x0 = x - (step_size + 0.5) * patch_size[1];  x1 = x + (step_size + 0.5) * patch_size[1]

                x0, x1, y0, y1 = int(np.round(x0)), int(np.round(x1)), int(np.round(y0)), int(np.round(y1))
                r0, r1, c0, c1 = int(r0), int(r1), int(c0), int(c1)

                #print(patch_size, pad_size)
                #print(f'r{r}, c{c}, [{r0}, {r1}], [{c0}, {c1}],         y{y}, x{x}, [{y0}, {y1}], [{x0}, {x1}]')
                
                input_img = Image.fromarray(img_array[y0:y1, x0:x1, :])
                input_img = input_img.resize([256, 256])
                input_x = preprocess(input_img).unsqueeze(0).to(device)
                image_features = model.encode_image(input_x)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = logit_scale * image_features @ text_features.t()
                similarity = similarity.squeeze(0).to('cpu').numpy()
                #cv2.imshow(str(similarity), np.array(input_img))
                #cv2.waitKey()
                # Add to accumulate
                for i in range(similarity.shape[0]):
                    times[i, r0:r1, c0:c1] += 1
                    scores[i, r0:r1, c0:c1] += similarity[i]
    
    # Gray segmentation
    #gray_map = split_gray_img( c_mean_shift(np.array(img)), n_labels=6 )

    # CLIP segmentation
    segmentations = []
    seg_images = []
    heatmaps = []
    for i in range(scores.shape[0]):
        scr = scores[i, :, :]
        tim = times[i, :, :]
        scr = np.divide(scr, tim)
        scr = scr.reshape(-1)
        #scr[np.argsort(scr)[-10:]] = 100
        #scr[np.argsort(scr)[:-10]] = 0
        #scr = scr.reshape(sample_size)
        #scr = gaussian_filter(scr, sigma=1)
        #scr = scr.reshape(-1)
        ind = np.argsort(scr)[-2-topk:-2]
        segment = np.zeros(scr.shape)
        segment[ind] = 1
        segment = segment.reshape(sample_size)

        indices = np.where(segment == 1)
        indices = np.concatenate(indices, axis=0)
        indices = indices.reshape(2, -1).T
        indices = indices.tolist()
        segmentations.append({'object': objects[i], 'segment': segment, 'indices': indices})

        #segment = scr.reshape(sample_size) / 100
        #segment = cv2.resize(segment, tuple(img_size[::-1]), interpolation=cv2.INTER_NEAREST)
        #gray_segment = gray_map[segment == 1]
        #maxima = np.argsort( np.bincount(gray_segment) )[-1]
        #gray_segment = np.zeros(img_size)
        #segment[gray_map != maxima] = 0

        #gray = Image.fromarray(gray_map).convert('RGB')
        segment = cv2.resize(segment, tuple(img_size[::-1]), interpolation=cv2.INTER_NEAREST)
        buff = np.zeros([img_size[0], img_size[1], 3])
        buff[:, :, 0] = segment * 255 * 0.5
        seg_img = np.clip(np.array(img) + buff, 0, 255).astype(np.uint8)
        #seg_img = np.clip(0.6 * np.array(img) + 0.4 * np.expand_dims(seg_img, axis=2), 0, 255).astype(np.uint8)
        seg_img = Image.fromarray(seg_img)
        

        fig, ax = plt.subplots()
        ax.imshow(scr.reshape(sample_size), cmap='hot', interpolation='nearest')

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        heatmap = Image.open(buf)

        seg_images.append(seg_img)
        heatmaps.append(heatmap)
    
    return segmentations, seg_images, heatmaps


def get_strokes_from_clip(clip_model, device, image, text_prompt, colors):
    clip_preprocess = T.Compose([T.Resize([224, 224]),
                                 T.ToTensor(),
                                 T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

    objects = parse_caption(text_prompt, list(colors.keys()))
    segments, _, _ = search_objects(clip_model, device, clip_preprocess, 
                                                image, list(objects.keys()), sample_size=[16, 16], step_size=1.2, topk=2)

    # Generate prior
    strokes = []
    for obj in range(len(segments)):
        ind = segments[obj]['indices']
        col_name = objects[segments[obj]['object']]
        for r, c in ind:
            r = r*16;  c = c*16
            color = colors[col_name]
            strokes.append({'index': [r, c], 'color': color})
    
    return strokes