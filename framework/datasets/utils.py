import numpy as np
import torch
from PIL import Image



def rgb_to_gray(x):
    return 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]


def output_to_pil(x):
    x = x.detach()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    if x.is_cuda:
        x = x.cpu()
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    if x.shape[2] == 1:
        x = x[:, :, 0]
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x
