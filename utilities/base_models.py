#! /usr/bin/env python

import numpy as np

import torch
import torchvision
from torch import nn

from tqdm import tqdm

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
        
def apply_to_loader(model, data_loader, num_dim):
    pbar = tqdm(data_loader, miniters = 5, leave = True)

    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")

    data_enc = np.empty((len(data_loader.dataset), num_dim), dtype = 'f4')
    data_lbl = np.empty(len(data_loader.dataset), dtype = 'i4')

    idx = 0

    for batch_idx, (data, target) in enumerate(pbar):

        num_smpl = data.size(0)

        data = data.to(device)
        enc = model(data)

        data_enc[idx : idx + num_smpl] = enc.detach().cpu().numpy()
        data_lbl[idx : idx + num_smpl] = target.detach().cpu().numpy()

        idx += num_smpl
        
    return data_enc, data_lbl

class Encoder_Faces(nn.Module):
    def __init__(self, num_feat, num_dim):
        super().__init__()
        self.num_feat = num_feat
        self.num_dim = num_dim

        self.cnn_encode_layers = nn.Sequential(
            nn.Conv2d(3, self.num_feat * 4, kernel_size = 5, padding = 0, stride = 2, bias = False),
            nn.BatchNorm2d(self.num_feat * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.num_feat * 4, self.num_feat * 3, kernel_size = 3, padding = 1, stride = 1, bias = False),
            nn.BatchNorm2d(self.num_feat * 3),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.num_feat * 3, self.num_feat * 2, kernel_size = 3, padding = 0, stride = 2, bias = False),
            nn.BatchNorm2d(self.num_feat * 2),
            nn.LeakyReLU(0.2, inplace=True))

        self.linear_encode_layers = nn.Sequential(
            nn.Linear(self.num_feat * 2 * 14 * 14, self.num_dim))

        self.linear_encode_layers.apply(init_weights)

    def forward(self, x):
        x = self.cnn_encode_layers(x)
        x = x.view(-1, self.num_feat * 2 * 14 * 14)
        x = self.linear_encode_layers(x)

        return x
    
class Decoder_Faces(nn.Module):
    def __init__(self, num_feat, num_dim):
        super().__init__()
        self.num_feat = num_feat
        self.num_dim = num_dim

        self.linear_decode_layers = nn.Sequential(
            nn.Linear(self.num_dim, self.num_feat * 2 * 14 * 14),
            nn.ReLU(True))

        self.cnn_decode_layers = nn.Sequential(
            nn.ConvTranspose2d(self.num_feat * 2, self.num_feat * 3, kernel_size = 4, stride = 2, bias = False),
            nn.BatchNorm2d(self.num_feat * 3),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.num_feat * 3, self.num_feat * 4, kernel_size = 3, stride = 1, bias = False, padding = 1),
            nn.BatchNorm2d(self.num_feat * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.num_feat * 4, 3, kernel_size = 6, stride = 2, bias = False),
            nn.Sigmoid())

        self.linear_decode_layers.apply(init_weights)

    def forward(self, x):
        x = self.linear_decode_layers(x)
        x = x.view(-1, self.num_feat * 2, 14, 14)
        x = self.cnn_decode_layers(x)
        
        return x

class Discriminator_Faces(nn.Module):
    def __init__(self, num_dim):
        super().__init__()
        self.num_dim = num_dim
        
        self.discriminator = nn.Sequential(nn.Linear(self.num_dim, self.num_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.num_dim, self.num_dim//2),
                                           nn.ReLU(),
                                           nn.Linear(self.num_dim//2, 1))

        self.discriminator.apply(init_weights)
        
    def forward(self, x):
        return self.discriminator(x)
    
class Encoder_MNIST(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.encode = nn.Sequential(
                        nn.Linear(28 * 28, 256),
                        nn.LeakyReLU(0.2, True),
                        nn.Linear(256, 64),
                        nn.LeakyReLU(0.2, True),
                        nn.Linear(64, self.dim))
        
        self.encode.apply(init_weights)

    def forward(self, x):
        return self.encode(x.view(-1, 1 * 28 * 28))

class Decoder_MNIST(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.decode = nn.Sequential(
                        nn.Linear(self.dim, 64),
                        nn.ReLU(True),
                        nn.Linear(64, 256),
                        nn.ReLU(True),
                        nn.Linear(256, 28 * 28), 
                        nn.Sigmoid())

        self.decode.apply(init_weights)

    def forward(self, x):
        return self.decode(x).view(-1, 1, 28, 28)
    
class Discriminator_MNIST(nn.Module):
    def __init__(self, dim, num_labels):
        super().__init__()
        self.dim = dim
        self.num_labels = num_labels if (num_labels > 0) else -1

        self.discriminator = nn.Sequential(nn.Linear(self.dim + self.num_labels + 1, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, 1))

        self.discriminator.apply(init_weights)
        
    def forward(self, x):
        return self.discriminator(x)
    
