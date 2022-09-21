import os, sys
from PIL import Image
import numpy as np
import yaml
import cv2
import time
import math

from utils_func import *


def get_line(x1, y1, x2, y2):
    points = []
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2-y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append([y, x])
        else:
            points.append([x, y])
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points


class Stroke():
    
    def __init__(self, img_size, sample_size=[256, 256]):
        self.img_size = img_size
        self.sample_size = sample_size
        self.indices = []
        self.colors = []

    def add(self, points, color):
        # Project all coordinates to sample_size
        points = np.array(points)
        points = points / self.img_size * self.sample_size
        points = points.astype(int)

        scores = np.zeros([16, 16])
        ly, lx = points[0]
        for y, x in points[1:]:
            points = get_line(lx, ly, x, y)
            for px, py in points:
                scores[py//16, px//16] += 1
            ly, lx = y, x

        rows, cols = np.where(scores >= 12)

        for i in range(len(rows)):
            coord = [rows[i] * 16, cols[i] * 16]
            ind = self.search(coord)
            if ind != None:
                self.indices.pop(ind)
                self.colors.pop(ind)
            self.indices.append(coord)
            self.colors.append(color)
        
    
    def delete(self, coord):
        ind = self.search(coord)
        if ind != None:
            self.indices.pop(ind)
            self.colors.pop(ind)

    def search(self, coord):
        try:
            ind = self.indices.index(coord)
            return ind
        except ValueError as e:
            return None

    def get_strokes(self):
        strokes = []
        for i in range(self.__len__()):
            strokes.append({'index': self.indices[i], 'color': self.colors[i]})
        return strokes

    def __len__(self):
        return len(self.indices)



class RectRegion():
    def __init__(self, img_size, sample_size=[256, 256]):
        self.rects = []
        self.img_size = img_size
        self.sample_size = sample_size

    def add(self, rect):
        x0, y0, x1, y1 = rect

        if y0 > y1:
            temp = y0
            y0 = y1
            y1 = temp
        if x0 > x1:
            temp = x0
            x0 = x1
            x1 = temp

        x0 = math.floor( (x0 / self.img_size[1] * self.sample_size[1]) / 16 )
        y0 = math.floor( (y0 / self.img_size[0] * self.sample_size[0]) / 16 )
        x1 = math.ceil( (x1 / self.img_size[1] * self.sample_size[1]) / 16 )
        y1 = math.ceil( (y1 / self.img_size[0] * self.sample_size[0]) / 16 )

        select = True
        extend = []
        for i, r in enumerate(self.rects):
            if x0>=r[0] and y0>=r[1] and x1<=r[2] and y1<=r[3]:
                select = False
            if x0<=r[0] and y0<=r[1] and x1>=r[2] and y1>=r[3]:
                extend.append(i)

        if select:
            for i in sorted(extend, reverse=True):
                del self.rects[i]
            self.rects.append([x0, y0, x1, y1])
    
    def get_sample_indices(self):
        indices = []
        for [x0, y0, x1, y1] in self.rects:
            for y in range(y0, y1):
                for x in range(x0, x1):
                    indices.append([y, x])
        unique = []
        [unique.append(x) for x in indices if x not in unique]
        indices = sorted(unique)
        return indices
