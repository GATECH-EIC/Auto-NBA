"""prepare CIFAR and SVHN
"""

from __future__ import print_function

import sys
import os
import glob
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from autoaugment import ImageNetPolicy


crop_size = 32
padding = 4


class ImageNet_For_Search(Dataset):
    def __init__(self, root, transforms_=None, num_class=100):
        self.transform = transforms_
        self.files = []
        self.labels = []

        # dir_list = np.random.choice(os.listdir(os.path.join(root, mode)), num_class, replace=False)
        dir_list = os.listdir(root)[:num_class]
        for label, dirname in enumerate(dir_list):
            for fname in os.listdir(os.path.join(root, dirname)):
                assert 'JPEG' in fname or 'jpg' in fname or 'png' in fname
                self.files.append(os.path.join(root, dirname, fname))
                self.labels.append(label)

        data_list = list(zip(self.files, self.labels))
        np.random.shuffle(data_list)
        self.files, self.labels = zip(*data_list)

        self.files = list(self.files)
        self.labels = list(self.labels)


    def __getitem__(self, index):
        img = self.transform(Image.open(self.files[index % len(self.files)]).convert('RGB'))
        label = self.labels[index % len(self.files)]
        
        return img, label

    def __len__(self):
        # return max(len(self.files), len(self.files_B))
        return len(self.files)


def prepare_train_data_for_search(dataset='imagenet', datadir='/home/yf22/dataset', num_class=100):
    if 'imagenet' in dataset:
        train_dataset = ImageNet_For_Search(
            datadir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]), 
            num_class=num_class)

    else:
        train_dataset = None
        
    return train_dataset


def prepare_test_data_for_search(dataset='imagenet', datadir='/home/yf22/dataset', num_class=100):
    if 'imagenet' in dataset:
        train_dataset = ImageNet_For_Search(
            datadir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            ]), 
            num_class=num_class)

    else:
        train_dataset = None
        
    return train_dataset


def prepare_train_data_autoaugment(dataset='imagenet', datadir='/home/yf22/dataset'):
    if 'imagenet' in dataset:
        train_dataset = torchvision.datasets.ImageFolder(
            datadir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))

    else:
        train_dataset = None
        
    return train_dataset


def prepare_train_data(dataset='imagenet', datadir='/home/yf22/dataset'):
    if 'imagenet' in dataset:
        train_dataset = torchvision.datasets.ImageFolder(
            datadir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))

    else:
        train_dataset = None
        
    return train_dataset


def prepare_test_data(dataset='imagenet', datadir='/home/yf22/dataset'):

    if 'imagenet' in dataset:
        test_dataset = torchvision.datasets.ImageFolder(datadir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ]))

    else:
        test_dataset = None

    return test_dataset
