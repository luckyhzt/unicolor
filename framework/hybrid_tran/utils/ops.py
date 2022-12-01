import torch



def to_sequence(x):
    # Convert a tensor [b, c, w, h] into sequence [b, l, c]
    b, c, w, h = x.shape
    x = x.view(b, c, w*h)
    x = x.permute(0, 2, 1).contiguous()
    return x


def squared_euclidean_distance(a, b):
    b = torch.transpose(b, 0, 1)
    a2 = torch.sum(torch.square(a), dim=1, keepdims=True)
    b2 = torch.sum(torch.square(b), dim=0, keepdims=True)
    ab = torch.matmul(a, b)
    d = a2 - 2 * ab + b2
    return d


def quantize(x, centroids):
    b, c, h, w = x.shape
    # [B, C, H, W] => [B, H, W, C]
    x = x.permute(0, 2, 3, 1).contiguous()
    x = x.view(-1, c)  # flatten to pixels
    d = squared_euclidean_distance(x, centroids)
    x = torch.argmin(d, 1)
    x = x.view(b, h, w)
    return x


def unquantize(x, centroids):
    return centroids[x]


def expand_relative_position_bias(rel_bias):
    ''' 
    Input shape: [num_heads, 2 * seq_length]
    '''

    num_heads, length = rel_bias.shape
    length = length // 2

    # Now we have to shift in order to compute relative biases.
    # Example: length = 3
    # Say we want:  [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
    # Start: [[-2, -1, 0, 1, 2, 3], [-2, -1, 0, 1, 2, 3], [-2, -1, 0, 1, 2, 3]]
    # We linearize: [-2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3]
    # We slice: [-2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3, -2, -1, 0]
    # We reshape: [[-2, -1, 0, 1, 2], [3, -2, -1, 0, 1], [2, 3, -2, -1, 0]]
    # We slice: [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]

    rel_bias = rel_bias.repeat(1, length)  # [heads, 2*len^2]
    rel_bias = rel_bias[:, :length*(2*length - 1)]  # [heads, 2*len^2-len]
    rel_bias = rel_bias.reshape([num_heads, length, 2 * length - 1])  # [heads, len, 2*len-1]
    rel_bias = rel_bias[:, :, length - 1:]  # [heads, len, len]

    return rel_bias


'''def pixel_to_bins(x, bins_per_channel):
    c = x.shape[1]
    x = x.permute(0, 2, 3, 1).contiguous()  # [b, h, w, c]
    # Input shape: [batch, channel, height, width]
    if x.dtype == torch.float:
        x = x * 255.0
    elif x.dtype == torch.int:
        x.type(torch.float32)
    x = x / (256.0 / bins_per_channel)
    x = x.type(torch.int64).type(torch.float32)
    bins = [bins_per_channel**i for i in range(c-1, -1, -1)]
    bins = torch.FloatTensor(bins).to(x.device)
    x = torch.matmul(x, bins).type(torch.int64)
    return x


def bins_to_pixel(x, bins_per_channel, num_channel):
    bins = [bins_per_channel**i for i in range(num_channel-1, -1, -1)]

    pixels = []
    for i in range(num_channel):
        chn = torch.floor_divide(x, bins[i])
        pixels.append(chn)  # [b, h, w]
        x = torch.fmod(x, bins[i])

    pixels = torch.stack(pixels, dim=-1)  # [b, h, w, c] 
    pixels = pixels.permute(0, 3, 1, 2)  # [b, c, h, w]
    pixels = (256.0 / bins_per_channel) * pixels.type(torch.float32)
    pixels = pixels / 255.0
    return pixels'''




if __name__ == '__main__':
    a = 0
