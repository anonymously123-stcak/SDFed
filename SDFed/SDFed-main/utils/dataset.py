import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity
from functools import partial
from typing import Optional, Callable
from torch.utils.model_zoo import tqdm
import PIL
import tarfile

import os
import os.path
import logging
import torchvision.datasets.utils as utils
import pickle
import string
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset
from Dassl.dassl.utils import check_isfile

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

class Datum:
    def __init__(self, data, label=0, domain=0, classname=""):
        # assert isinstance(impath, str)
        # assert check_isfile(impath)

        self._data = data
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname

class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        data = mnist_dataobj.data
        target = mnist_dataobj.targets

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):

        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class FashionMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        mnist_dataobj = FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)
        data = mnist_dataobj.data
        target = mnist_dataobj.targets

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class SVHN_custom(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.train is True:
            svhn_dataobj = SVHN(self.root, 'train', self.transform, self.target_transform, self.download)
            data = svhn_dataobj.data
            target = svhn_dataobj.labels
        else:
            svhn_dataobj = SVHN(self.root, 'test', self.transform, self.target_transform, self.download)
            data = svhn_dataobj.data
            target = svhn_dataobj.labels

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target
    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

