#! /usr/bin/env python

import numpy as np

import torch
import torchvision

import PIL
import random
import warnings

warnings.simplefilter("ignore", category=UserWarning)


class RandomDatasetWrapper(torch.utils.data.Dataset):
    """Wrap an existing PyTorch Dataset from a pytorch tensor which chooses a random subset."""

    def __init__(self, input_dataset, size = -1, transform=None):
        """
        Args:
            input_dataset (Dataset): input Dataset
            size (size): Number of entries of the input dataset to choose
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.__input_dataset = input_dataset
        
        if (size > 0) and (size < 1):
            size = int(size * len(self.__input_dataset))
        
        if (size > len(self.__input_dataset)) or (size < 0):
            self.__size = len(self.__input_dataset)
        else:
            self.__size = size
            
        if self.__size == len(self.__input_dataset):
            self.__choice = list(range(len(self.__input_dataset)))
        else:
            self.__choice = random.sample(list(range(len(self.__input_dataset))), self.__size)
        
        self.__transform = transform

    def __len__(self):
        return self.__size

    def __getitem__(self, idx):
        sample, _ = self.__input_dataset[self.__choice[idx]]

        if self.__transform is not None:
            sample = self.__transform(sample)
        
        return sample, _

class MyTransform(object):
    """Wrapper class to apply one of the functional transforms."""
    def __init__(self, functional_transform, rng_min, rng_max):
        self.__rng_min, self.__rng_max = rng_min, rng_max
        self.__functional_transform = functional_transform
    
    def __call__(self, x):
        return self.__functional_transform(x, random.uniform(self.__rng_min, self.__rng_max))    
    
class RandomizeTransform(object):
    """Wrapper class to apply one of the functional transforms."""
    def __init__(self, transform, p = 0.5):
        self.__transform = transform
        self.__p = p
    
    def __call__(self, x):
        if self.__p < random.random():
            return x
        else:
            return self.__transform(x)
        
class ApplyRandomSubset(object):
    """Apply a randomly selected subset of a list of transformations

    Args:
        transforms (list or tuple): list of transformations
        size (integer): size of the subset
    """

    def __init__(self, transforms, size):
        self.__transforms = transforms
        self.__size = size

    def __call__(self, x):
        for t in random.sample(self.__transforms, self.__size):
            x = t(x)
        return x
    
def apply_transform(in_data, transform):
    out_data = torch.empty(in_data.size(), dtype = torch.float32)

    for i in range(in_data.size(0)):
        out_data[i] = transform(in_data[i])
        
    return out_data    
