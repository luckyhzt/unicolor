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
import io
import kornia
import torchvision
import skimage.color

from utils_func import *



if __name__ == '__main__':
    stroke_path = 'C:/MyFiles/CondTran/data/all_text_strokes.json'
    img_dir = 'D:\\colorization_results\\final_text_new_color\\images\\3'
    device = 'cuda:0'

    # Load CLIP
    clip_preprocess = T.Compose([T.Resize([224, 224]),
                                        T.ToTensor(),
                                        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
    clip_model, _ = clip.load("ViT-B/32", device=device)

    # Load strokes
    with open(stroke_path, 'r') as file:
        all_strokes = json.load(file)

    pbar = tqdm(enumerate(all_strokes))


    all_similarities = []

    with torch.no_grad():
        for i, file in pbar:
            # Prepare text features
            caption = file['caption']
            text = clip.tokenize(caption).to(device)
            text_features = clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = clip_model.logit_scale.exp()

            img_name = file['image'].split('.')[0]
            img = Image.open(os.path.join(img_dir, f'{img_name}.png')).resize([256, 256]).convert('RGB')

            x = clip_preprocess(img).unsqueeze(0).to(device)

            img_features = clip_model.encode_image(x)

            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            similarity = logit_scale * img_features @ text_features.t()
            similarity = similarity.squeeze(0).to('cpu').numpy()

            all_similarities.append(similarity)
    
    print(len(all_similarities))
    print(np.mean(all_similarities))
        