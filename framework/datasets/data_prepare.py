from multiprocessing import Pool
import glob
import os
import cv2
from functools import partial
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import json


def image_colorfulness(image):
	# split the image into its respective RGB components
	(R, G, B) = cv2.split(image.astype("float"))
	# compute rg = R - G
	rg = np.absolute(R - G)
	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)
	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
	# combine the mean and standard deviations
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)


def process_superpixel(image):
    slic = cv2.ximgproc.createSuperpixelSLIC(image)
    slic.iterate(5)

    # retrieve the segmentation result
    labels = slic.getLabels()
    quantize_img = image.copy()
    low = np.min(labels)
    high = np.max(labels)
    for c in range(low, high+1):
        quantize_img[labels == c, :] = np.mean(image[labels == c, :], axis=0)

    result = Image.fromarray(quantize_img)
    return result


def data_process(image_paths, data_root, save_path):
    img_size = [int(256*1.15), int(256*1.15)]
    
    infos = []
    for i in tqdm(range(len(image_paths))):
        path = image_paths[i]
        img = Image.open(path).resize(img_size).convert('RGB')
        img = np.array(img)
        # Colorfulness
        colorfulness = image_colorfulness(img)
        # SLIC superpixel
        slic = process_superpixel(img)
        save_name = os.path.relpath(path, data_root)
        save_name = save_name.split('.')[0] + '.png'
        save_name = os.path.join(save_path, save_name)
        os.makedirs(os.path.split(save_name)[0], exist_ok=True)
        slic.save(save_name)

        dataset_root = os.path.join(data_root, os.path.pardir)
        infos.append({
            'image_path': os.path.relpath(path, dataset_root), 
            'slic_path': os.path.relpath(save_name, dataset_root), 
            'colorfulness': colorfulness
        })
    return infos



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str, help='The root directories of the dataset split.')
    parser.add_argument('dataset', type=str, help='"coco" or "imagenet"')
    parser.add_argument('--num_process', type=int, default=8)
    args = parser.parse_args()

    assert args.dataset in ('coco', 'imagenet')
    if args.dataset == 'coco':
        data_roots = [
            os.path.join(args.data_root, 'train2017'),
            os.path.join(args.data_root, 'unlabeled2017'),
            os.path.join(args.data_root, 'val2017'),
        ]
    elif args.dataset == 'imagenet':
        data_roots = [
            os.path.join(args.data_root, 'train'),
            os.path.join(args.data_root, 'val'),
        ]

    annotations = []
    for data_root in data_roots:
        save_path = data_root + '_slic'
        
        image_exts = ('*.jpg', '*.png', '*.JPEG')
        files = []
        for exts in image_exts:
            files.extend( glob.glob(os.path.join(data_root, exts)) )
        print('Processing:', data_root, 'Total images:', len(files))
    
        sub_files = []
        nums = len(files) // args.num_process
        for n in range(args.num_process):
            if n < args.num_process - 1:
                sub_files.append(files[n*nums:(n+1)*nums])
            else:
                sub_files.append(files[n*nums:])
        
        pool = Pool(args.num_process)
        info_lists = pool.map(partial(data_process, data_root=data_root, save_path=save_path), sub_files)
        infos = sum(info_lists, [])

        with open(data_root+'_meta.json', 'w') as f:
            json.dump(infos, f)
        
        