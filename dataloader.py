# dataloader.py

import math

import torch
import datasets
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import os
import utils as utils

class Dataloader:

    def __init__(self, args):
        self.args = args

        self.loader_input = args.loader_input
        self.loader_label = args.loader_label
        self.prefetch = args.prefetch

        self.split_test = args.split_test
        self.split_train = args.split_train
        self.dataset_test_name = args.dataset_test
        self.dataset_train_name = args.dataset_train
        self.resolution = (args.resolution_wide, args.resolution_high)

        self.input_filename_test = args.input_filename_test
        self.label_filename_test = args.label_filename_test
        self.input_filename_train = args.input_filename_train
        self.label_filename_train = args.label_filename_train

        if self.dataset_train_name == 'LSUN':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(root=args.dataroot, classes=['bedroom_train'],
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_train_name == 'CASIA':
            self.dataset_train = datasets.ImageFolder(root=self.args.dataroot,
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ])
                )

        elif self.dataset_train_name == 'RECONSTRUCTION':
            self.dataset_train = datasets.RECONSTRUCTION(root=self.args.dataroot,
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]), nc=args.nchannels
                )

        elif self.dataset_train_name == 'CELEBA':
            self.dataset_train = datasets.ImageFolder(root=self.args.dataroot + "/train", # change it back to train before training
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.RandomHorizontalFlip(),
                    # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.01),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ])
                )

        elif self.dataset_train_name == 'SSFF':
            self.dataset_train = datasets.ImageFolder(root=self.args.dataroot + "/train",
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ])
                )

        elif self.dataset_train_name == 'CIFAR10' or self.dataset_train_name == 'CIFAR100':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(root=self.args.dataroot, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.01),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
                )

        elif self.dataset_train_name == 'GMM':
            self.dataset_train = datasets.GMM(args)

        elif self.dataset_train_name == 'GMMRing':
            self.dataset_train = datasets.GMM_Ring(args)

        elif self.dataset_train_name == 'CocoCaption' or self.dataset_train_name == 'CocoDetection':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(root=self.args.dataroot, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_train_name == 'STL10' or self.dataset_train_name == 'SVHN':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(root=self.args.dataroot, split='train', download=True,
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_train_name == 'MNIST':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(root=self.args.dataroot, train=True, download=True,
                transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
                )

        elif self.dataset_train_name == 'ImageNet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            # self.dataset_train = datasets.ImageFolder(root=os.path.join(self.args.dataroot, "train"),
            #     transform=transforms.Compose([
            #         transforms.Scale(self.resolution),
            #         transforms.CenterCrop(self.resolution),
            #         transforms.RandomHorizontalFlip(),
            #         transforms.ToTensor(),
            #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            #         normalize,
            #        ])
            #     )
            self.dataset_train = getattr(datasets, self.dataset_train_name)(root=self.args.dataroot,
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    normalize
                   ]),
                prefetch = self.args.prefetch
                )

        elif self.dataset_train_name == 'FRGC':
            self.dataset_train = datasets.ImageFolder(root=self.args.dataroot+self.args.input_filename_train,
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ])
                )

        elif self.dataset_train_name == 'Folder':
            self.dataset_train = datasets.ImageFolder(root=self.args.dataroot+self.args.input_filename_train,
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_train_name == 'FileList':
            self.dataset_train = datasets.FileList(self.input_filename_train, self.label_filename_train, self.split_train,
                self.split_test, train=True,
                transform_train=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                transform_test=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                loader_input=self.loader_input,
                loader_label=self.loader_label,
                )

        elif self.dataset_train_name == 'FolderList':
            self.dataset_train = datasets.FileList(self.input_filename_train, self.label_filename_train, self.split_train,
                self.split_test, train=True,
                transform_train=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                transform_test=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                loader_input=self.loader_input,
                loader_label=self.loader_label,
                )

        else:
            raise(Exception("Unknown Dataset"))

        if self.dataset_test_name == 'LSUN':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(root=args.dataroot, classes=['bedroom_val'],
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_test_name == 'CIFAR10' or self.dataset_test_name == 'CIFAR100':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(root=self.args.dataroot, train=False, download=True,
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_test_name == 'CELEBA':
            self.dataset_test = datasets.ImageFolder(root=self.args.dataroot + "/test",
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ])
                )

        elif self.dataset_test_name == 'RECONSTRUCTION':
            pass

        elif self.dataset_test_name == 'SSFF':
            self.dataset_test = datasets.ImageFolder(root=self.args.dataroot + "/test",
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ])
                )

        elif self.dataset_test_name == 'CocoCaption' or self.dataset_test_name == 'CocoDetection':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(root=self.args.dataroot, train=False, download=True,
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_test_name == 'STL10' or self.dataset_test_name == 'SVHN':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(root=self.args.dataroot, split='test', download=True,
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_test_name == 'MNIST':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(root=self.args.dataroot, train=False, download=True,
                transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
                )

        elif self.dataset_test_name == 'ImageNet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            self.dataset_test = getattr(datasets, self.dataset_test_name)(root=self.args.dataroot,
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    normalize
                   ])
                )

        elif self.dataset_test_name == 'FRGC':
            self.dataset_test = datasets.ImageFolder(root=self.args.dataroot+self.args.input_filename_test,
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ])
                )

        elif self.dataset_test_name == 'Folder':
            self.dataset_test = datasets.ImageFolder(root=self.args.dataroot+self.args.input_filename_test,
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_test_name == 'FileList':
            self.dataset_test = datasets.FileList(self.input_filename_test, self.label_filename_test, self.split_train,
                self.split_test, train=True,
                transform_train=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                loader_input=self.loader_input,
                loader_label=self.loader_label,
                )

        elif self.dataset_test_name == 'FolderList':
            self.dataset_test = datasets.FileList(self.input_filename_test, self.label_filename_test, self.split_train,
                self.split_test, train=True,
                transform_train=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                loader_input=self.loader_input,
                loader_label=self.loader_label,
                )
        elif self.dataset_test_name == 'GMM':
            self.dataset_test = datasets.GMM(args)

        elif self.dataset_test_name == 'GMMRing':
            self.dataset_test = datasets.GMM_Ring(args)

        elif self.dataset_test_name is None:
            pass

        else:
            raise(Exception("Unknown Dataset"))

    def create(self, flag=None, shuffle=True):
        if flag == "Train":
            dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.args.batch_size,
                shuffle=shuffle, num_workers=int(self.args.nthreads), pin_memory=True)
            return dataloader_train

        if flag == "Test":
            dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.args.batch_size,
                shuffle=shuffle, num_workers=int(self.args.nthreads), pin_memory=True)
            return dataloader_test

        if flag == None:
            dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.args.batch_size,
                shuffle=shuffle, num_workers=int(self.args.nthreads), pin_memory=True)

            dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.args.batch_size,
                shuffle=shuffle, num_workers=int(self.args.nthreads), pin_memory=True)
            return dataloader_train, dataloader_test
