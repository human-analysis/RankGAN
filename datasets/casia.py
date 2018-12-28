import torch.utils.data as data
from PIL import Image
import os

class CASIA(data.Dataset):
    def __init__(self, dataroot, transforms=None):
        self.dataroot = dataroot
        self.transforms = transforms
        self.objects = []

        for _, folder, _ in os.walk(dataroot):
            for _, _, files in os.walk("{}/{}".format(dataroot, folder[0])):
                for file in files:
                    image = Image.open("{}/{}/{}".format(dataroot, folder[0], file))
                break
            break
