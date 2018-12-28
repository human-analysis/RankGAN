import numpy as np
import os
import torch
import json
from PIL import Image
import torch.utils.data as data
import pickle
import torchvision.transforms as transforms


class CELEBA(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.objects = []
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        with open("{}/celeba.pkl".format(self.root), 'rb') as f_stream:
            self.objects = pickle.load(f_stream)

        self.length = len(self.objects)
        print("Total = {}".format(self.length))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
            target contains the cropped objects, camera and vehicle parameters
        """
        image = self.objects[index]
        image = image.float() / 255
        image = self.normalize(image)
        return image, 1

    def __len__(self):
        return self.length
