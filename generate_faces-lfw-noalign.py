#! /usr/bin/env python3

import numpy as np

import torch
import torchvision

from matplotlib import pyplot as plt
import seaborn as sns

def get_dataset(data_path, transform):
    return torchvision.datasets.ImageFolder(root=data_path, transform=transform)

def get_loader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size = 32, num_workers = 0, shuffle = True)

def image_qc(img):
    fig, ax  = plt.subplots(figsize=(12, 12), nrows = 1, ncols = 1)

    ax.imshow(torchvision.utils.make_grid(img.detach(), padding = 8, nrow = 8).permute(1, 2, 0))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.show()
   

use_grayscale = False
im_size = 64
data_path = './data/lfw'

if use_grayscale:
    transform=torchvision.transforms.Compose([torchvision.transforms.Resize(im_size, 3), torchvision.transforms.Grayscale(num_output_channels=1), torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambda x : x * 255)])
else:
    transform=torchvision.transforms.Compose([torchvision.transforms.Resize(im_size, 3), torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambda x : x * 255)])
    
                                         
dataset = get_dataset(data_path, transform)


only_one = set()
idx_get = set()

for i in range(len(dataset)):
    print("{} / {}".format(i, len(dataset)))
    
    data = dataset[i]
    target = data[1]
    
    if target not in only_one:
        only_one.add(target)
        idx_get.add(i)
        
print(len(only_one))

data_raw = np.memmap("./data/LFWfaces_noalign_onlyone_dataset_{}x{}x{}x{}_uint8.bin".format(len(only_one), 1 if use_grayscale else 3, im_size, im_size), shape = (len(only_one), 1 if use_grayscale else 3, im_size, im_size), dtype = np.uint8, mode = 'w+')
target_raw = np.memmap("./data/LFWfaces_noalign_onlyone_target_uint32.bin", shape = len(only_one), dtype = np.uint32, mode = 'w+')

j = 0
for i in sorted(idx_get):
    print("{} / {}".format(j, len(idx_get)))
    
    data = dataset[i]

    data_raw[j] = data[0]
    target_raw[j] = data[1]
    
    j += 1
    
    
    