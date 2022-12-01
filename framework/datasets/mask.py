import torch
from torch import nn
import numpy as np
import random


class Mask_Generator(nn.Module):
    def __init__(self, input_shape, mode_prob={'random': 0.92, 'block': 0.0, 'full': 0.08}, strokes=16):
        super().__init__()
        self.input_shape = input_shape
        self.seq_len = input_shape[0] * input_shape[1]
        self.mode_prob = mode_prob
        self.strokes = strokes

    def forward(self):
        mask = np.ones(self.seq_len).astype(int)

        # Select masked indices
        mode = random.choices(list(self.mode_prob.keys()), weights=list(self.mode_prob.values()))[0]
        if mode == 'random':
            n = np.random.randint(low=self.seq_len//16, high=self.seq_len)
            masked_ind = np.arange(0, self.seq_len)
            masked_ind = np.random.permutation(masked_ind)
            masked_ind = masked_ind[:n]

        elif mode == 'block':
            masked_ind = np.arange(0, self.seq_len)
            unmask = select_region(self.input_shape, np.random.randint(low=2, high=16))
            for u in unmask:
                masked_ind = masked_ind[masked_ind != u]

        elif mode == 'full':
            masked_ind = np.arange(0, self.seq_len)
        masked_ind = masked_ind.astype(int)
        # Select cond indices
        strokes = min(self.strokes, len(masked_ind))
        cond_ind = np.random.permutation(masked_ind)[:strokes]
        # Generate mask
        mask[masked_ind] = 0
        mask[cond_ind] = -1
        mask = mask.reshape(self.input_shape)

        return mask


def select_region(input_shape, num):
    size = np.sqrt(num * 2)
    size = int(size)
    
    indices = np.arange(input_shape[0]*input_shape[1])
    indices = indices.reshape(input_shape)

    x0 = np.random.randint(low=0, high=input_shape[0] - size + 1)
    x1 = x0 + size
    y0 = np.random.randint(low=0, high=input_shape[1] - size + 1)
    y1 = y0 + size

    indices = indices[y0:y1, x0:x1]
    indices = indices.reshape(-1)

    indices = np.random.choice(indices, size=num, replace=False)

    return indices


if __name__ == '__main__':
    gen = Mask_Generator([4, 4])
    x = torch.zeros(2, 4, 4, 8).to('cuda:0')
    masked_indices = gen(x)
    print(masked_indices)