from __future__ import print_function

import argparse
import glob
import os, sys
import time
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform_lib
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

import lib.TestTransforms as transforms
from models.ColorVidNet import ColorVidNet
from models.FrameColor import warp_color
from models.NonlocalNet import VGG19_pytorch, WarpNet
from utils.util import (batch_lab2rgb_transpose_mc, folder2vid, mkdir_if_not,
                        save_frames, tensor_lab2rgb, uncenter_l)
from utils.util_distortion import CenterPad, Normalize, RGB2Lab, ToTensor


class ImageWarper():
    def __init__(self, device):
        current_path = os.path.dirname(os.path.realpath(__file__))
        self.device = device
        self.nonlocal_net = WarpNet(1).eval().to(device).requires_grad_(False)
        self.vggnet = VGG19_pytorch().eval().to(device).requires_grad_(False)
        vggnet_path = os.path.join(current_path, 'data', 'vgg19_conv.pth')
        self.vggnet.load_state_dict(torch.load(vggnet_path))
        nonlocal_test_path = os.path.join(current_path, 'checkpoints', 'video_moredata_l1', 'nonlocal_net_iter_76000.pth')
        self.nonlocal_net.load_state_dict(torch.load(nonlocal_test_path))

    def warp_image(self, image, image_ref):
        image_size = [216 * 2, 384 * 2]
        transform = transforms.Compose(
            [T.Resize(image_size), RGB2Lab(), ToTensor(), Normalize()]
        )

        IB_lab_large = transform(image_ref).unsqueeze(0).to(self.device)
        IB_lab = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear")

        IA_lab_large = transform(image).unsqueeze(0).to(self.device)
        IA_lab = torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear")
        IA_l = IA_lab[:, 0:1, :, :]
        IA_ab = IA_lab[:, 1:3, :, :]

        # start the frame colorization
        with torch.no_grad():
            nonlocal_BA_lab, similarity_map, features_A_gray = warp_color(IA_l, IB_lab, self.vggnet, self.nonlocal_net, temperature=0.01)
            I_current_ab_predict = nonlocal_BA_lab[:, 1:3, :, :]

        # upsampling
        curr_bs_l = IA_lab_large[:, 0:1, :, :]
        curr_predict = (
            torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear") * 1.25
        )
        '''curr_bs_l = nonlocal_BA_lab[:, 0:1, :, :]
        curr_predict = I_current_ab_predict.data.cpu()'''

        IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict[:32, ...])

        image = np.clip(IA_predict_rgb, 0, 255).astype(np.uint8)
        image = Image.fromarray(image)

        similarity_map = similarity_map.squeeze(0).squeeze(0).to('cpu').numpy()

        return image, similarity_map



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda:0')

    parser.add_argument("--img_path", type=str, default="C:/MyFiles/colorization_transformer/imgs/12.jpg")
    parser.add_argument("--ref_path", type=str, default="C:/MyFiles/colorization_transformer/imgs/13.jpg",)

    opt = parser.parse_args()

    warper = ImageWarper(opt.device)
    warped_image, similarity_map = warper.warp_image(opt.img_path, opt.ref_path)

    similarity_map = cv2.resize(similarity_map, (16, 16), interpolation=cv2.INTER_NEAREST)

    print(np.argsort(similarity_map.reshape(-1))[-10:])
    warped_image.show()
