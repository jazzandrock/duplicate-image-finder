import os
import re

from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache
from itertools import combinations, product
from operator import itemgetter, attrgetter

from PIL import Image
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Find duplicate images')
parser.add_argument('--path', metavar='path', type=str, required=True,
                    help='path to images folder')
args = parser.parse_args()
    

images_folder = args.path

class MyImage:
    def __init__(self, img_name):
        path = os.path.join(images_folder, img_name)
        img = Image.open(path)
        
        self.img = np.array(img)
        self.name = img_name
        self.number = int(re.findall(r'\d+', img_name)[0])
        
        self.describe = lru_cache(maxsize=None)(self.describe)
    
    def __str__(self): 
        return f'img_{self.name}'
    
    def __repr__(self):
        return str(self)
    
    def describe(self, descriptor_class):
        return descriptor_class(self)
    
    def show(self):
        pass

class Descriptor(ABC):
    def __init__(self, img):
        pass
    
    @abstractmethod
    def similar(self, other):
        pass
    
    @abstractmethod
    def visualize(self):
        pass

class ColorHistogramDescriptor(Descriptor):
    def __init__(self, img):
        self.img = img
        
        descriptor = np.zeros(shape=(3, 255), dtype=np.float)
        
        for i in range(3):
            channel = img.img[:, :, i]
            channel_descriptor = np.histogram(channel, bins=np.arange(256), density=True)[0]
            descriptor[i] = channel_descriptor
            
        self.hist = descriptor
        
    def similar(self, other):
        return sum(
            np.linalg.norm(self.hist[i] - other.hist[i], ord=1)
            for i in range(3)
        )
    
    def visualize(self):
        pass

images_names = os.listdir(images_folder)
images = list(map(MyImage, images_names))

def similarity(descriptor_class, img1, img2):
    d1 = img1.describe(descriptor_class)
    d2 = img2.describe(descriptor_class)
    return d1.similar(d2)

threshold = 1.0

for im1, im2 in combinations(images, r=2):
    if similarity(ColorHistogramDescriptor, im1, im2) < threshold:
        print(f'{im1.name} {im2.name}')
