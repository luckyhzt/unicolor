import sys
import os
import cv2
from matplotlib import patches
from numpy.core.numeric import indices
import torch
import numpy as np
import yaml
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageOps
import json
import nltk
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from skimage import color


def load_model(model, dir, step):
    # Load config
    config_path = os.path.join(dir, 'config.yaml')
    with open(config_path, 'rb') as fin:
        config = yaml.safe_load(fin)
    model_config = config['model']
    model_config['learning_rate'] = 0.0
    # Load model
    loaded = model.load_from_checkpoint(
        find(dir, step+'.ckpt'),
        **model_config,
        load_vqgan_from_separate_file=False,
        strict=True
    )
    return loaded
        
def output_to_pil(x):
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.to('cpu')
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    if x.shape[2] == 1:
        x = x[:, :, 0]
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def find(path, name):
    for root, dirs, files in os.walk(path):
        for f in files:
            if name in f:
                return os.path.join(root, f)

def save_result(html, index, images, texts=None):
    save_dir = html.get_image_dir()
    ims, txts, links = [], [], []
    for i, img in enumerate(images):
        os.makedirs(os.path.join(save_dir, f'{i}'), exist_ok=True)
        img.save(os.path.join(save_dir, f'{i}/{index}.png'))
        # Save in html file
        ims.append(f'{i}/{index}.png')
        if texts == None or i >= len(texts):
            txts.append('')
        else:
            txts.append(texts[i])
        links.append('')
    html.add_images(ims, txts, links)

def preprocess(img, size):
    if size == None:
        size = [(img.size[1] // 16) * 16, (img.size[0] // 16) * 16]
    transform = T.Compose([T.Resize(size), T.ToTensor()])
    x = transform(img)
    x = x * 2 - 1
    x = x.unsqueeze(0)
    return x

def get_indices_from_segment(segment):
    segment = np.array(segment)
    indices = np.where(segment == 1)
    indices = np.concatenate(indices, axis=0)
    indices = indices.reshape(2, -1).T
    return indices.tolist()

def draw_color_sk(l, c, rect):
    y0, y1, x0, x1 = rect
    l = np.array(l)
    lab = color.rgb2lab(l)
    l = lab[:, :, 0:1]
    ab = lab[:, :, 1:3]
    draw = np.array(c)
    if len(draw.shape) == 1:
        draw = np.expand_dims(draw, axis=[0, 1])
    draw = color.rgb2lab(draw)
    ab[y0:y1, x0:x1, :] = draw[:, :, 1:3]
    lab = np.concatenate([l, ab], axis=2)
    img = color.lab2rgb(lab)
    img = (img*255).astype(np.uint8)
    return Image.fromarray(img)

def draw_color(l, color, rect):
    y0, y1, x0, x1 = rect
    l = np.array(l.convert('RGB'))
    lab = cv2.cvtColor(l, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0:1]
    ab = lab[:, :, 1:3]
    draw = np.array(color).astype(np.uint8)
    if len(draw.shape) == 1:
        draw = np.expand_dims(draw, axis=[0, 1])
    draw = cv2.cvtColor(draw, cv2.COLOR_RGB2LAB)
    ab[y0:y1, x0:x1, :] = draw[:, :, 1:3]
    lab = np.concatenate([l, ab], axis=2)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img)

def draw_full_color(l, color, rect):
    y0, y1, x0, x1 = rect
    img = np.array(l.convert('RGB'))
    draw = np.array(color).astype(np.uint8)
    if len(draw.shape) == 1:
        draw = np.expand_dims(draw, axis=[0, 1])
    img[y0:y1, x0:x1, :] = draw[:, :, :]
    return Image.fromarray(img)

def get_sample_range(length, sample_size):
    num = int( np.ceil( float(length) / float(sample_size) ) )
    start = sample_size // 2
    stop = start + (num - 1) * sample_size
    steps = np.linspace(start=start, stop=stop, num=num).astype(int)
    return list(steps)

def get_input_range(rows, cols, r, c, sample_shape):
    # Index range for input
    c0 = c - sample_shape[1] // 2
    c1 = c0 + sample_shape[1]
    if c0 < 0:
        c0 = 0
        c1 = c0 + sample_shape[1]
    if c1 > cols:
        c1 = cols
        c0 = c1 - sample_shape[1]
    r0 = r - sample_shape[0] // 2
    r1 = r0 + sample_shape[0]
    if r0 < 0:
        r0 = 0
        r1 = r0 + sample_shape[0]
    if r1 > rows:
        r1 = rows
        r0 = r1 - sample_shape[0]
    
    return int(r0), int(c0), int(r1), int(c1)

def get_mask_range(rows, cols, r, c, mask_size, input_range):
    # Index range for mask
    r0 = r - mask_size[0] // 2;  r1 = r0 + mask_size[0]
    c0 = c - mask_size[1] // 2;  c1 = c0 + mask_size[1]
    r0 = max(0, r0);  r1 = min(rows, r1)
    c0 = max(0, c0);  c1 = min(cols, c1)
    r_start = input_range[0];  c_start = input_range[1]
    return int(r0 - r_start), int(c0 - c_start), int(r1 - r_start), int(c1 - c_start)

def get_predict_range(rows, cols, r, c, patch_size, input_range):
    # Index range for predict patch
    r0 = r - patch_size[0] // 2;  r1 = r0 + patch_size[0]
    c0 = c - patch_size[1] // 2;  c1 = c0 + patch_size[1]
    r0 = max(0, r0);  r1 = min(rows, r1)
    c0 = max(0, c0);  c1 = min(cols, c1)

    pos = []
    for row in range(r0, r1):
        for col in range(c0, c1):
            nrow = row - input_range[0]
            ncol = col - input_range[1]
            width = input_range[3] - input_range[1]
            pos.append( int(nrow * width + ncol) )
    
    return int(r0), int(c0), int(r1), int(c1), pos


def color_resize(l, color):
    color = color.resize(l.size)
    resized = draw_color(l.convert('RGB'), color, [None, None, None, None])
    return resized


def draw_strokes(image, img_size, strokes):
    org_size = image.size[::-1]
    # Draw strokes
    draw_img = image.copy().convert('RGB')
    for stk in strokes:
        ind = stk['index']
        ind = [int(ind[0] / img_size[0] * org_size[0]), int(ind[1] / img_size[1] * org_size[1])]
        patch_size = [int(org_size[0]/img_size[0]*10), int(org_size[1]/img_size[1]*10)]
        border_size = [int(org_size[0]/img_size[0]*3), int(org_size[1]/img_size[1]*3)]
        color = np.zeros(patch_size+[3])
        color[:, :] = stk['color']
        color = color.astype(np.uint8)
        color = cv2.copyMakeBorder(color, border_size[0], border_size[0], border_size[1], border_size[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        draw_img = draw_full_color(draw_img, color, [ind[0], ind[0]+color.shape[0], ind[1], ind[1]+color.shape[1]])
    return draw_img.resize(org_size[::-1])


def limit_size(img, minsize, maxsize):
    if img.size[0] < minsize or img.size[1] < minsize:
        if img.size[0] < img.size[1]:
            size = [minsize, minsize*(img.size[1]/img.size[0])]
        else:
            size = [minsize*(img.size[0]/img.size[1]), minsize]
        img = img.resize([int(size[0]), int(size[1])])

    elif img.size[0] > maxsize or img.size[1] > maxsize:
        if img.size[0] > img.size[1]:
            size = [maxsize, maxsize*(img.size[1]/img.size[0])]
        else:
            size = [maxsize*(img.size[0]/img.size[1]), maxsize]
        img = img.resize([int(size[0]), int(size[1])])
        
    return img

def replace(words, ori, rep):
    for i in range(len(words)):
        if words[i] == ori:
            words[i] = rep
    return words

def visualize_mask(img, mask):
    img = np.array(img)
    mask = np.expand_dims(np.array(mask), axis=2)
    mask_img = np.clip(0.6*img + 0.4*mask, 0, 255).astype(np.uint8)
    return Image.fromarray(mask_img)



class Drawboard():
    def __init__(self, img, grid_size=[16, 16]):
        self.grid_size = grid_size
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'color_table.yaml'), 'r') as fin:
            self.colors = yaml.safe_load(fin)
        # Show image
        self.img = np.array(img)
        fig, self.ax = plt.subplots()
        self.ax.set_xticks(np.arange(0, 257, self.grid_size[1]))
        self.ax.set_yticks(np.arange(0, 257, self.grid_size[0]))
        self.ax.grid(alpha=0.5)
        self.plot = self.ax.imshow(self.img)
        # Color selection buttons
        gap = 0.95 / len(self.colors)
        position = [0.0, 0.0, 0.1, gap-0.01]
        self.buttons = []
        for i in range(len(self.colors)):
            name = list(self.colors.keys())[i]
            col = self.colors[name]
            col = np.array(col).astype(np.float32) / 255.0
            col = list(col)
            self.buttons.append( Button(plt.axes(position), name, color=col) )
            self.buttons[i].on_clicked( lambda x, text=name: self.select(x, text) )
            position[1] += gap
        # Draw on image
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.selected_color = 'red'
        self.clicked = {}
        self.stroke = {}
    
    def select(self, event, text):
        self.selected_color = text

    def onclick(self, event):
        if event.inaxes not in [self.ax]:
            return
        x, y = [int(event.xdata)//self.grid_size[1]*self.grid_size[1], int(event.ydata)//self.grid_size[0]*self.grid_size[0]]
        if self.selected_color not in list(self.clicked.keys()):
            self.clicked[self.selected_color] = []
        self.clicked[self.selected_color].append([y, x])

        '''x, y = [int(event.xdata // 8), int(event.ydata // 8)]
        if self.selected_color not in list(self.stroke.keys()):
            self.stroke[self.selected_color] = []
        self.stroke[self.selected_color].append([y, x])'''
        x0 = x
        y0 = y
        x1 = x0 + self.grid_size[1]
        y1 = y0 + self.grid_size[0]
        self.img = draw_color(self.img, self.colors[self.selected_color], [y0, y1, x0, x1])
        self.img = np.array(self.img)
        self.plot.set_data(self.img)
        plt.draw()