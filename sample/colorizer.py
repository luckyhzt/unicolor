import enum
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
from ImageMatch.warp import ImageWarper



class Colorizer():
    def __init__(self, ckpt_file, device, img_size, load_clip=False, load_warper=False):
        self.ckpt_path = os.path.abspath( os.path.join(ckpt_file, os.pardir) )
        self.ckpt_file = (ckpt_file.split('/')[-1]).split('.')[0]
        self.model_path = os.path.abspath( os.path.join(self.ckpt_path, os.pardir, os.pardir) )
        self.device = device
        self.img_size = img_size
        self.sample_size = [self.img_size[0] // 16, self.img_size[1] // 16]
        self.load_model(load_clip, load_warper)

    
    def load_model(self, load_clip, load_warper):
        # Load CLIP
        if load_clip:
            self.clip_preprocess = T.Compose([T.Resize([224, 224]),
                                        T.ToTensor(),
                                        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
            self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'color_table.yaml'), 'r') as fin:
                self.color_table = yaml.safe_load(fin)
        # Load image warper
        if load_warper:
            self.image_warper = ImageWarper(self.device)
        # Load colorizer
        os.chdir(self.model_path)
        sys.path.append(self.model_path)
        module = importlib.import_module('hybrid_tran.models.colorization')
        model = getattr(module, 'Colorization')
        self.transformer = load_model(model, self.ckpt_path, self.ckpt_file).to(self.device).eval().requires_grad_(False)


    @torch.no_grad()
    def sample(self, image, strokes, topk, prior_image=None, mask_indices=None, sample_indices=None, progress=None):
        image = image.convert('L')
        x_gray = preprocess(image, self.img_size).to(self.device)

        if len(strokes) > 0:
            cond = []
            cond_indices = []
            for stk in strokes:
                ind = stk['index']
                ind = torch.Tensor([0, ind[0]//16, ind[1]//16]).long().to(self.device)
                color = stk['color']
                color = torch.Tensor(color).to(self.device)
                color = color / 255.0 * 2.0 - 1.0
                cond.append(color.unsqueeze(0))
                cond_indices.append(ind.unsqueeze(0))
            cond = torch.cat(cond, axis=0)
            cond_indices = torch.cat(cond_indices, axis=0)
        else:
            cond = None
            cond_indices = None

        if prior_image == None:
            _, f_gray = self.transformer.chroma_vqgan.encode(None, x_gray)
            rows, cols = f_gray.shape[2:4]
            color_idx = self.transformer.mask_token * torch.ones([1, rows, cols]).to(f_gray.device).long()
        else:
            prior_image = prior_image.convert('RGB')
            x_prior = preprocess(prior_image, self.img_size).to(self.device)
            color_idx, f_gray = self.transformer.chroma_vqgan.encode(x_prior, x_gray)
            rows, cols = f_gray.shape[2:4]
            color_idx = color_idx.reshape(color_idx.shape[0], rows, cols)
        
        # Mask indices
        if mask_indices != None:
            for [r, c] in mask_indices:
                color_idx[:, r, c] = self.transformer.mask_token

        # Sample indices
        if sample_indices == None:
            sample_indices = []
            for r in range(rows):
                for c in range(cols):
                    sample_indices.append([r, c])

        for i, [r, c] in enumerate(sample_indices):
            # Input gray feature
            cond_gray = f_gray.clone()
            cond_gray = cond_gray.reshape(cond_gray.shape[0], cond_gray.shape[1], -1)
            cond_gray = cond_gray.permute(0, 2, 1).contiguous()
            # Input color indices
            idx = color_idx.clone()

            logits = self.transformer.hybrid_tran(idx, cond_gray, cond, cond_indices)
            logits = logits.view(logits.shape[0], rows, cols, -1)
            logits = logits[:, r, c, :]
            logits = logits.reshape(-1, logits.shape[-1])
            logits = self.transformer.top_k_logits(logits, topk)
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            color_idx[:, r, c] = ix

            if progress is not None:
                progress.emit(int(100 * (i+1) / len(sample_indices)))

        gen = self.transformer.chroma_vqgan.decode(color_idx, f_gray)
        gen = output_to_pil(gen[0])
        gen = color_resize(image, gen)
        return gen

    
    @torch.no_grad()
    def upsample(self, gray, color, topk=1, patch_size=[1, 1], mask_size=[1, 1], progress=None):
        img_size = [(gray.size[0] // 16) * 16, (gray.size[1] // 16) * 16]
        gray = gray.convert('L').resize(img_size)
        color = color.resize(img_size)

        x_gray = preprocess(gray, None).to(self.device)
        x_color = preprocess(color, None).to(self.device)
        color_idx, f_gray = self.transformer.chroma_vqgan.encode(x_color, x_gray)
        color_idx = color_idx.reshape(1, f_gray.shape[2], f_gray.shape[3])

        # Sampling parameters
        B = f_gray.shape[0]
        sample_shape = [16, 16]
        rows, cols = f_gray.shape[2:4]
        rrange = get_sample_range(rows, patch_size[0])
        crange = get_sample_range(cols, patch_size[1])

        # Start upsampling
        total = len(rrange) * len(crange)
        i = 0
        for r in rrange:
            for c in crange:
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
                idx[:, rm0:rm1, cm0:cm1] = self.transformer.mask_token

                logits = self.transformer.hybrid_tran(idx, cond, None, None)
                logits = logits[:, pos, :]
                logits = logits.reshape(-1, logits.shape[-1])
                logits = self.transformer.top_k_logits(logits, topk)
                probs = F.softmax(logits, dim=-1)
                ix = torch.multinomial(probs, num_samples=1)
                ix = ix.reshape(B, rp1-rp0, cp1-cp0)
                color_idx[:, rp0:rp1, cp0:cp1] = ix
                # Send progress
                i += 1
                if progress is not None:
                    progress.emit(int(100 * (i+1) / total))

        gen = self.transformer.chroma_vqgan.decode(color_idx, f_gray)
        gen = output_to_pil(gen[0])
        gen = color_resize(gray, gen)
        return gen


    def get_strokes_from_clip(self, image, text_prompt):
        image = image.convert('L').convert('RGB')
        objects = self.parse_caption(text_prompt)
        segments, heatmaps = self.search_objects(image, list(objects.keys()), sample_size=[16, 16], step_size=1.2, topk=2)
        # Generate strokes
        strokes = []
        for obj in range(len(segments)):
            ind = segments[obj]['indices']
            col_name = objects[segments[obj]['object']]
            for r, c in ind:
                r = r*16;  c = c*16
                color = self.color_table[col_name]
                strokes.append({'index': [r, c], 'color': color})
    
        return strokes, heatmaps
    

    @torch.no_grad()
    def get_strokes_from_exemplar(self, image, exemplar_image):
        image = image.convert('L').convert('RGB')
        warped_img, similarity_map = self.image_warper.warp_image(image, exemplar_image)
        warped = warped_img.copy()
        warped_img = warped_img.resize(self.img_size)

        similarity_map = cv2.resize(similarity_map, tuple(self.sample_size))
        similarity_map = similarity_map.reshape(-1)
        threshold = min(0.23, np.sort(similarity_map)[-5])
        indices = np.where( (similarity_map >= threshold))

        strokes = []
        warped_img = np.array(warped_img)
        for ind in indices[0]:
            index = [ind//16 * 16, ind%16 * 16]
            color = warped_img[index[0]:index[0]+16, index[1]:index[1]+16, :]
            color = color.mean(axis=(0, 1))
            strokes.append({'index': index, 'color': color.tolist()})
        
        return strokes, warped


    @torch.no_grad()
    def search_objects(self, img, objects, sample_size=[16, 16], step_size=2.5, topk=2):
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
            text = clip.tokenize(objects).to(self.device)
            text_features = self.clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.clip_model.logit_scale.exp()
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
                    input_x = self.clip_preprocess(input_img).unsqueeze(0).to(self.device)
                    image_features = self.clip_model.encode_image(input_x)
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
            #segment = cv2.resize(segment, tuple(img_size[::-1]), interpolation=cv2.INTER_NEAREST)
            #buff = np.zeros([img_size[0], img_size[1], 3])
            #buff[:, :, 0] = segment * 255 * 0.5
            #seg_img = np.clip(np.array(img) + buff, 0, 255).astype(np.uint8)
            #seg_img = np.clip(0.6 * np.array(img) + 0.4 * np.expand_dims(seg_img, axis=2), 0, 255).astype(np.uint8)
            #seg_img = Image.fromarray(seg_img)
            

            #fig, ax = plt.subplots()
            #ax.imshow(scr.reshape(sample_size), cmap='hot', interpolation='nearest')

            #buf = io.BytesIO()
            #fig.savefig(buf)
            #buf.seek(0)
            #heatmap = Image.open(buf)

            #seg_images.append(seg_img)
            #heatmaps.append(heatmap)

            mi = np.min(scr)
            ma = np.max(scr)
            heatmap = (scr - mi) / (ma - mi) * 255
            heatmap = heatmap.astype(np.uint8)
            heatmap = heatmap.reshape(sample_size)
            heatmap = cv2.resize(heatmap, tuple(img_size[::-1]), interpolation=cv2.INTER_NEAREST)

            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.addWeighted(heatmap, 0.6, np.array(img), 0.4, 0)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            heatmaps.append(Image.fromarray(heatmap))
        
        return segmentations, heatmaps

    def parse_caption(self, text_prompt):
        words = nltk.tokenize.word_tokenize(text_prompt)
        # replace grey with gray
        words = replace(words, 'grey', 'gray')
        objects = {}
        for col in self.color_table:
            if col in words:
                pos = words.index(col)
                if pos+1 < len(words):
                    objects[words[pos+1]] = col
        return objects


# Testing whether this class works
if __name__ == '__main__':
    ckpt_file = '/home/huangzhitong/code/unicolor/framework/checkpoints/unicolor_mscoco/mscoco_step259999'
    device = 'cuda:7'
    colorizer = Colorizer(ckpt_file, device, [256, 256], load_clip=True, load_warper=True)
