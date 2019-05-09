# loaders.py

import torch
import numpy as np
import scipy.io as sio
from skimage import io
from PIL import Image
import cv2 as cv


def loader_skimage(path):
    return io.imread(path)


def loader_cvimage(path):
    image = cv.imread(path, cv.IMREAD_COLOR) # this reads every image as if it is colored.
    image = image[:, :, [2,1,0]] # BGR to RGB
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image


def loader_image(path):
    return Image.open(path).convert('RGB')


def loader_torch(path):
    return torch.load(path)


def loader_numpy(path):
    return np.load(path)


def loader_mat(path):
    return sio.loadmat(path)

def loader_landmark(path):
    # read landmark data in the CMUCars dataset
    with open(path) as f:
        numInstance = int(next(f).rstrip('\r\n'))
        # occlusion = np.ndarray((numInstance, 20, 1))
        landmarks = np.ndarray((numInstance, 20, 2), dtype='int')
        for inst in range(numInstance):
            pose = int(next(f).rstrip('\r\n'))
            label = np.array([[int(x) for x in next(f).rstrip('\r\n').split('\t')] for i in range(20)])
            # occlusion[inst,] = label[:, 0:1]
            landmarks[inst,] = label[:, 1:3]
    return landmarks#, occlusion
