import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as T
import matplotlib.pyplot as plt

import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)

def conv3x3(in_channels, out_channels, stride=1):
    """3x3 kernel size with padding convolutional layer in ResNet BasicBlock."""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)