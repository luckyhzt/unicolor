import numpy as np
import skimage.color
from PIL import Image
import os
from sklearn import *
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

from utils_func import *
from html_images import *



def getfeatures(img, segm=None, stepsize=7, use_loc = False):
    """
    getfeatures - extract features from an image
     [X, Y, L] = getfeatures(img, stepsize)
     INPUT
     img      - the image   [type = np.ndarray or Image]
     segm     - the gt segmentation (optional)
     stepsize - window stepsize
     OUTPUT
      X   - the features: each row is a feature vector  [type = np.ndarray]
      Y   - the GT segment labels for each feature (if segm is provided)
      L   - dictionary containing location information of each feature
    """
    winsize = 5 # ensure it is an odd number
    if stepsize > winsize:
        raise Exception('stepsize larger than window size')
    # convert to LAB
    yimg = skimage.color.rgb2lab(img)

    offset = np.floor((winsize-1)/2)
    sy,sx, sc = img.shape

    if use_loc:
        Xdim = 5
    else:
        Xdim = 3
    
    # extract window patches with stepsize
    patches = skimage.util.view_as_windows(yimg, (winsize, winsize, 3), step=stepsize)    
    psize = patches.shape
    
    # get coordinates of windows
    rangex = np.arange(psize[1])*stepsize + offset
    rangey = np.arange(psize[0])*stepsize + offset

    X = np.zeros((psize[0] * psize[1], Xdim));
    
    if segm is None:
        Y = None
    else:
        Y = np.zeros((X.shape[0],))
    
    i = 0
    for x in range(psize[1]):
        for y in range(psize[0]):
            myl = np.mean(patches[y,x,0,:,:,0].flatten())
            myu = np.mean(patches[y,x,0,:,:,1].flatten())
            myv = np.mean(patches[y,x,0,:,:,2].flatten())
            myy = int(rangey[y])
            myx = int(rangex[x])
            
            if use_loc:
                X[i,:] = [myl, myu, myv, myx, myy]
            else:
                X[i,:] = [myl, myu, myv]
                
            if Y is not None:
                Y[i] = segm[myy, myx]
                
            i = i + 1 
    
    L = {'rangex':rangex, 'rangey':rangey, 'offset':offset, 'sx':sx, 'sy':sy, \
         'stepsize':stepsize, 'winsize':winsize}
    return X, Y, L

def labels2seg(Y,L):
    """
    labels2segm - form a segmentation image using cluster labels
    segm = labels2segm(Y, L)
    Y - cluster labels for each location
    L - location dictionary from getfeatures
    segm - output segmentation image
    """
    segm = np.zeros((L['sy'], L['sx']))
    # <= offset if floor((winsize-1)/2)>= floor(stepsize/2) 
    rstep = int(np.floor(L['stepsize']/2.0)) 
    stepbox = range(-rstep, L['stepsize'] - rstep)
    rx = np.asarray(L['rangex'], dtype=int) + int(L['offset'])
    ry = np.asarray(L['rangey'], dtype=int) + int(L['offset'])
    Y_reshaped = Y.reshape((ry.size, rx.size),order='F')
    for i in stepbox:
        for j in stepbox:
            segm[np.ix_(ry + j, rx + i)] = Y_reshaped
    ## Now fil in the borders if they are missing
    minx = min(rx) + stepbox[0] - 1
    maxx = max(rx) + stepbox[-1] + 1
    miny = min(ry) + stepbox[0] - 1
    maxy = max(ry) + stepbox[-1] + 1

    if 0 <= minx:
        ## fill in left edge
        segm[:, 0:minx+1] = segm[:,minx+1].reshape((-1,1))
    if maxx < L['sx']:
        ## fill in right edge
        segm[:,maxx:] = segm[:,maxx-1].reshape((-1,1))
    if 0 < miny:
        ## fill in top edge
        segm[0:miny+1,:] = segm[miny+1,:].reshape((1,-1))
    if maxy < L['sy']:
        ## fill in bottom edge
        segm[maxy:,:] = segm[maxy-1,:].reshape((1,-1))
    return segm    

def colorsegms(segm, img):
    """
    colorsegm - color a segmentation based on the image
    csegm = colorsegm(segm, img)
    segm = the segmentation image  [type = np.ndarray]
    img = the original image    [type = np.ndarray (or Image)]
    csegm = the colored segmentation -- each segment is colored based on the 
            average pixel color within the segment.
    """
    img = np.asarray(img).copy()
    if segm.shape[0:2] != img.shape[0:2]:
        raise Exception('The shape of segmentation and image are not consistent') 
    rimg, gimg, bimg = img[:,:,0], img[:,:,1], img[:,:,2]
    for i in range(0, int(max(segm.flatten())) + 1):
        # assume label starts from 1
        ind = (segm == i)
        rimg[ind] = np.mean(rimg[ind].flatten())
        gimg[ind] = np.mean(gimg[ind].flatten())
        bimg[ind] = np.mean(bimg[ind].flatten())
    # handle outliers from DBSCAN
    ind = (segm == -1)
    rimg[ind] = 0
    gimg[ind] = 0
    bimg[ind] = 0
    return img


np.random.seed(100)

if __name__ == '__main__':
    img_dir = 'C:\\MyFiles\\Dataset\\coco\\val2017'
    save_dir = 'C:\\Users\\lucky\\Desktop\\generated_strokes_coco'

    images = os.listdir(img_dir)
    html = HTML(save_dir, 'Sample')

    all_strokes = []
    for i, name in tqdm(enumerate(images)):
        I_color = Image.open(os.path.join(img_dir, name)).convert('RGB').resize([256, 256])
        I_gray = I_color.convert('L').convert('RGB')
        img = np.array(I_color)

        # extract features, each row is a feature vector
        Xo, _, L = getfeatures(img, stepsize=4, use_loc=True)

        # normalize features
        scaler = preprocessing.StandardScaler()  
        X = scaler.fit_transform(Xo)
        X[:, 0] *= 1

        # Mean Shift
        bw = 1.0
        ms = cluster.MeanShift(bandwidth=bw, bin_seeding=True)
        Y = ms.fit_predict(X)

        # convert cluster labels to a segmentation image
        segm = labels2seg(Y, L)
        
        block_size = 16
        stroke = []
        for r in range(0, 256, 16):
            for c in range(0, 256, 16):
                r0 = np.clip(r + 8 - block_size//2, 0, 255)
                r1 = np.clip(r + 8 + block_size//2, 0, 255)
                c0 = np.clip(c + 8 - block_size//2, 0, 255)
                c1 = np.clip(c + 8 + block_size//2, 0, 255)
                pixels = segm[r0:r1, c0:c1].reshape(-1).astype(int)
                if np.bincount(pixels).max() == (r1 - r0) * (c1 - c0):
                    stroke.append([r, c])
        
        if len(stroke) > 0:
            strokes = []
            for r, c in stroke:
                color = img[r:r+16, c:c+16, :]
                color = color.reshape(-1, 3)
                color = color.mean(axis=0)
                color = list(color)
                strokes.append({'index': [r, c], 'color': color})
                #l = draw_color(l, color, [r, r+16, c, c+16])
        
            all_strokes.append({'image': name, 'strokes': strokes})
        
        images = []
        images.append(I_color)
        images.append(draw_strokes(I_gray.convert('RGB'), [256, 256], strokes))
        save_result(html, index=i, images=images)
        html.save()
        
    with open(os.path.join(save_dir, 'coco_strokes.json'), 'w') as file:
        json.dump(all_strokes, file)

    


