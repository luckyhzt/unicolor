import os, sys

sys.path.append('./sample')
from PIL import Image
import numpy as np
import yaml
import importlib

from PyQt5 import QtWidgets,QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog

from ui import Ui_main

import cv2
import clip
import json

from utils_func import *
from sample_func import *
from clip_segment import *
from colorizer import Colorizer


class SampleThread(QtCore.QThread):
    message = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)
    result = QtCore.pyqtSignal(object, bool)

    def __init__(self, parent, source, mode, topk, num_samples):
        super(SampleThread, self).__init__()
        self.parent = parent
        self.source = source
        self.mode = mode
        self.topk = topk
        self.img_size = self.parent.img_size.copy()
        self.num_samples = num_samples
    
    def run(self):
        os.makedirs(os.path.join('demo', 'results'), exist_ok=True)
        os.makedirs(os.path.join('demo', 'results', 'stroke_cond'), exist_ok=True)
        os.makedirs(os.path.join('demo', 'results', 'text_cond'), exist_ok=True)
        os.makedirs(os.path.join('demo', 'results', 'exemplar_cond'), exist_ok=True)

        I_gray = self.parent.input_image.copy()
        I_gray.save(os.path.join('demo', 'results', 'gray.png'))

        all_strokes = []

        self.message.emit('Generating hint points...')

        if 'stroke' in self.mode:
            if self.source == 'input':
                strokes = self.parent.input_strokes.get_strokes()
            elif self.source == 'output':
                strokes = self.parent.output_strokes.get_strokes()
            all_strokes += strokes
            (self.parent.get_pixmap_image(self.source)).save(os.path.join('demo', 'results', 'stroke_cond', 'strokes.png'))
            save_strokes(I_gray, strokes, self.img_size, os.path.join('demo', 'results', 'stroke_cond'))
        if 'text' in self.mode:
            text_prompt = self.parent.text_input.toPlainText()
            strokes, heatmaps = self.parent.colorizer.get_strokes_from_clip(I_gray, text_prompt)
            all_strokes += strokes
            save_strokes(I_gray, strokes, self.img_size, os.path.join('demo', 'results', 'text_cond'))
            with open(os.path.join('demo', 'results', 'text_cond', 'prompt.txt'), 'w') as f:
                f.write(text_prompt)
            for i, h in enumerate(heatmaps):
                h.save(os.path.join('demo', 'results', 'text_cond', f'h{i}.png'))
        if 'exemplar' in self.mode:
            exemplar_image = self.parent.exemplar_image
            strokes, warped = self.parent.colorizer.get_strokes_from_exemplar(I_gray, exemplar_image)
            all_strokes += strokes
            warped = color_resize(I_gray, warped)
            warped.save(os.path.join('demo', 'results', 'exemplar_cond', f'warped.png'))
            exemplar_image.save(os.path.join('demo', 'results', 'exemplar_cond', 'reference.png'))
            save_strokes(I_gray, strokes, self.img_size, os.path.join('demo', 'results', 'exemplar_cond'))
        # Delete duplicate strokes
        strokes = []
        for stk in all_strokes:
            for s in strokes:
                if stk['index'] == s['index']:
                    break
            else:
                strokes.append(stk)

        save_strokes(I_gray, strokes, self.img_size, os.path.join('demo', 'results'))

        gen_imgs = []
        if self.source == 'input':
            for i in range(self.num_samples):
                self.message.emit(f'Sampling... {i+1}/{self.num_samples}')
                gen = self.parent.colorizer.sample(I_gray, strokes, self.topk, progress=self.progress)
                gen.save(os.path.join('demo', 'results', f'colorized_{i}.png'))
                gen_imgs.append(gen)

        if self.source == 'output':
            self.message.emit('Sampling... 1/1')
            (self.parent.get_pixmap_image('output')).save(os.path.join('demo', 'results', 'selected_reginons.png'))
            resample_indices = self.parent.output_regions.get_sample_indices()
            prior = self.parent.output_image.copy()
            prior.save(os.path.join('demo', 'results', 'original.png'))
            gen = self.parent.colorizer.sample(I_gray, strokes, self.topk, 
                prior_image=prior, mask_indices=resample_indices, sample_indices=resample_indices, progress=self.progress)
            gen.save(os.path.join('demo', 'results', f'recolorized.png'))
            gen_imgs.append(gen)

        '''if self.parent.is_upsample.isChecked():
            gen = self.parent.colorizer.upsample(I_gray, gen, progress=self.progress)
            gen.save(os.path.join('demo', 'results', 'upsampled.png'))'''

        self.message.emit('Complete.')
        self.progress.emit(0)
        self.parent.sample_btn.setEnabled(True)

        for img in gen_imgs:
            self.result.emit(img, False)
    

def save_strokes(I_gray, strokes, img_size, path):
    for stk in strokes:
        stk['index'] = np.int32(stk['index']).tolist()
        stk['color'] = np.int32(stk['color']).tolist()
    draw_img = draw_strokes(I_gray, img_size, strokes)
    draw_img.save(os.path.join(path, 'points.png'))
    with open(os.path.join(path, 'strokes.json'), 'w') as f:
        json.dump(strokes, f)



class LoadThread(QtCore.QThread):
    message = QtCore.pyqtSignal(str)

    def __init__(self, parent, path):
        super(LoadThread, self).__init__()
        self.parent = parent
        self.path = path
        self.curdir = os.getcwd()
    
    def run(self):
        self.message.emit(f'Loading models... ')
        self.parent.colorizer = Colorizer(self.path, self.parent.device, self.parent.img_size, load_clip=True, load_warper=True)
        os.chdir(self.curdir)
        self.message.emit(f'Models loaded.')
        #self.parent.model_info.setText(f'Model: {self.path}')