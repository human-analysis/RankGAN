import numpy as np
import os
import torch
import json
from PIL import Image
import torch.utils.data as data
import pickle
import torchvision.transforms as transforms
import utils
import copy
from utils import writetextfile, readtextfile
from datasets import loaders

class ImageNet(data.Dataset):
    def __init__(self, root, transform=None, prefetch=False):
        """This class is used to load images from the ImageNet Dataset as background images for CMUCars.

        :param images_list: list of images to be loaded.
        :param labels_list: list of labels to be loaded.
        :param categories: list [category] contains the name of a category.
        If set to -1, all categories are selected: [-1].
        :param transform: an AffineCropNGenerateHeatmaptransform must be used.
        :param prefetch: prefetch data in the memory to improve performance.
        :param make_partial_blockage: boolean to make some partial blockage for
               testing purposes.
        """

        self.root = root
        self.prefetch = prefetch
        self.transform = transform

        data_folder = os.path.join(self.root, 'Data/256/train/')
        # images_list = os.path.join(self.root, 'ImageSets/CLS-LOC/train_cls.txt')

        self.images_list = []
        for _, folders, _ in os.walk(data_folder):
            for folder in folders:
                for _, _, files in os.walk("{}/{}".format(data_folder, folder)):
                    for file in files:
                        self.images_list.append(os.path.join(data_folder, folder, file))
                    break
            break

        # self.images_list = [os.path.join(data_folder, '{}.JPEG'.format(x.split(' ')[0].rstrip('\n'))) for x in readtextfile(images_list)]
        # self.len = len(self.images_list)//10
        # self.images_list = self.images_list[:self.len]
        # if len(self.images_list) == 0:
        #     raise (RuntimeError("No images found"))
        self.len = len(self.images_list)

        if self.prefetch:
            # cache data to improve performance
            self.dataset = []
            for idx in range(len(self.images_list)):
                self.dataset.append(ImageNet.fetch_item(self.images_list[idx]))
        # self.__getitem__(0)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.prefetch:
            image = self.dataset[idx]
        else:
            image = ImageNet.fetch_item(self.images_list[idx])

        if self.transform:
            image = self.transform(image)
        if image.shape[0] == 1:
            # print(self.images_list[idx])
            image = torch.cat((image, image, image), 0)

        return image, 1

    @staticmethod
    def fetch_item(image_filename):
        # img = None
        # with Image.open(image_filename) as image:
        #     img = copy.deepcopy(image)
        image = loaders.loader_cvimage(image_filename)
        return Image.fromarray(image)
