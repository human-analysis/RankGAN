import numpy as np
import os
import torch
import json
from PIL import Image
import torch.utils.data as data
import pickle
import torchvision.transforms as transforms


class CELEBA(data.Dataset):
    def __init__(self, root, transform=None, prefetch=False):
        self.root = root
        self.transform = transform
        self.objects = []
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.prefetch = prefetch
        i = 0

        if self.prefetch:
            for _, folders, _ in os.walk(self.root):
                folder = folders[0]
                for _, _, files in os.walk("{}/{}".format(self.root, folder)):
                    for file in files:
                        with Image.open("{}/{}/{}".format(self.root, folder, file)) as image:
                            image_trans = transform(image)
                            image_trans = np.array(image_trans)
                            im_tensor = torch.from_numpy(image_trans.transpose((2, 0, 1)))
                            self.objects.append(im_tensor)
                            i += 1
                            print("Image {} added".format(i))
                    break
                break
        else:
            for _, folders, _ in os.walk(self.root):
                folder = folders[0]
                for _, _, files in os.walk("{}/{}".format(self.root, folder)):
                    for file in files:
                        filename = "{}/{}/{}".format(self.root, folder, file)
                        self.objects.append(filename)
                    break
                break

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
        if self.prefetch:
            image = self.objects[index]
        else:
            with Image.open(self.objects[index]) as image:
                image = self.transform(image)
            image = np.array(image)
            image = torch.from_numpy(image.transpose((2, 0, 1)))
        image = image.float() / 255
        image = self.normalize(image)
        return image, 1

    def __len__(self):
        return self.length
