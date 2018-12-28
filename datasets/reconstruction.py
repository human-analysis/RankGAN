import numpy as np
import os
import torch
import json
from PIL import Image
import torch.utils.data as data
import pickle
import torchvision.transforms as transforms


class RECONSTRUCTION(data.Dataset):
    def __init__(self, root, transform=None, nc=3):
        self.root = root
        self.transform = transform
        self.objects = []

        path, dirs, files = os.walk(root).__next__()
        # stages = ['stage1', 'stage2', 'stage3', 'wgan']
        stages = ['stage1', 'stage2', 'stage3']
        file_count = int(len(files)/(len(stages)+1))                  # 8 for reconstruction

        for file_i in range(file_count):
            prefix = files[(len(stages)+1)*file_i].split("_")[0]
            im_real = Image.open("{}/{}_Original.png".format(path, prefix))
            im_real_trans = transform(im_real)
            im_real.close()
            im_stages = [Image.open("{}/{}_Reconstructed_Stage_{}.png".format(path, prefix, k+1)) for k in range(len(stages))]     # Reconstructed for reconstruciton
            im_stages_trans = [transform(im_stages[k]) for k in range(len(stages))]
            im_closed = [im_stages[k].close() for k in range(len(stages))]
            if nc == 1:
                im_real_trans = togray(im_real_trans)
                im_stages_trans = [togray(im_stages_trans[j]) for j in range(len(stages))]
            # self.objects.append((im_real_trans, im_stages_trans[0], im_stages_trans[1], im_stages_trans[2], im_stages_trans[3]))
            self.objects.append((im_real_trans, im_stages_trans[0], im_stages_trans[1], im_stages_trans[2]))


        # for i in range(128, file_count+128):
        #     im_real = Image.open("{}/{}_Original.png".format(path, i))
        #     im_real_trans = transform(im_real)
        #     im_real.close()
        #     im_stages = [Image.open("{}/{}_Reconstructed_Stage_{}.png".format(path, i, k)) for k in range(1, 5)]     # Reconstructed for reconstruciton
        #     im_stages_trans = [transform(im_stages[k]) for k in range(4)]
        #     im_closed = [im_stages[k].close() for k in range(4)]
        #     if nc == 1:
        #         im_real_trans = togray(im_real_trans)
        #         im_stages_trans = [togray(im_stages_trans[j]) for j in range(4)]
        #     self.objects.append((im_real_trans, im_stages_trans[0], im_stages_trans[1], im_stages_trans[2], im_stages_trans[3]))

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

        return self.objects[index]

    def __len__(self):
        return min(self.length, 50000)

def togray(image):
    return 0.2989*image[0:1] + 0.5870*image[1:2] + 0.1140*image[2:3]
