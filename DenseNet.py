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
from utils import conv3x3, flatten, Flatten

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        num_classes = 10
        self.in_channel = 3
        self.channel1 = 32
        self.channel2 = 32
        self.channel3 = 64
        self.channel4 = 128
        self.channel5 = 256
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(self.in_channel, self.channel1, kernel_size=3, stride=1, padding=1, bias=False)#32x32
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.conv2 = nn.Conv2d(self.channel1, self.channel2, kernel_size=3, stride=1, padding=1, bias=False)#32x32
        self.bn2 = nn.BatchNorm2d(self.channel2)
        self.conv3 = nn.Conv2d(self.channel2, self.channel3, kernel_size=3, stride=2, padding=1, bias=False)#16x16
        self.bn3 = nn.BatchNorm2d(self.channel3)
        self.conv4 = nn.Conv2d(self.channel3, self.channel4, kernel_size=3, stride=2, padding=1, bias=False)#8x8
        self.bn4 = nn.BatchNorm2d(self.channel4)
        self.conv5 = nn.Conv2d(self.channel4, self.channel5, kernel_size=3, stride=2, padding=1, bias=False)#4x4
        self.bn5 = nn.BatchNorm2d(self.channel5)
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)
        self.fc = nn.Linear(256, num_classes)
        ##downsamples
        self.downsamplex1 = self.get_downsample(self.in_channel, self.channel1)
        self.downsamplex2 = self.get_downsample(self.in_channel, self.channel2)
        self.downsamplex3 = self.get_downsample(self.in_channel, self.channel3, stride=2)

        self.downsample13 = self.get_downsample(self.channel1, self.channel3, stride=2)
        self.downsample23 = self.get_downsample(self.channel2, self.channel3, stride=2)

        self.downsamplex4 = self.get_downsample(self.in_channel, self.channel4, stride=2, repeat=2)
        self.downsample14 = self.get_downsample(self.channel1, self.channel4, stride=2, repeat=2)
        self.downsample24 = self.get_downsample(self.channel2, self.channel4, stride=2, repeat=2)
        self.downsample34 = self.get_downsample(self.channel3, self.channel4, stride=2)

        self.downsamplex5 = self.get_downsample(self.in_channel, self.channel5, stride=2, repeat=3)
        self.downsample15 = self.get_downsample(self.channel1, self.channel5, stride=2, repeat=3)
        self.downsample25 = self.get_downsample(self.channel2, self.channel5, stride=2, repeat=3)
        self.downsample35 = self.get_downsample(self.channel3, self.channel5, stride=2, repeat=2)
        self.downsample45 = self.get_downsample(self.channel4, self.channel5, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def get_downsample(self, in_channel, out_channel, stride=1, repeat=1):
        downsample_layers = list()
        for _ in range(repeat):
            if (stride != 1) or (in_channel != out_channel):
                downsample = nn.Sequential(
                    conv3x3(in_channel, out_channel, stride=stride),
                    nn.BatchNorm2d(num_features=out_channel)
                )
                in_channel = out_channel
                downsample_layers.append(downsample)
        return nn.Sequential(*downsample_layers)

    def forward(self, x):
        """Forward pass of ResNet."""
        layer1 = self.conv1(x)
        layer1 = self.bn1(layer1)
        layer1 = self.relu(layer1)
        
        #layer1 = layer1 + self.downsamplex1(x)
        res1 = layer1 + self.downsamplex1(x)
        ########
        layer2 = self.conv2(res1)
        layer2 = self.bn2(layer2)
        layer2 = self.relu(layer2)
        
        #layer2 = layer2 + self.downsamplex2(x)
        #layer2 = layer2 + layer1

        res2 = layer2 + self.downsamplex2(x) + res1
        #########
        layer3 = self.conv3(res2)
        layer3 = self.bn3(layer3)
        layer3 = self.relu(layer3)
        
        #layer3 = layer3 + self.downsamplex3(x)
        #layer3 = layer3 + self.downsample13(layer1)
        #layer3 = layer3 + self.downsample23(layer2)

        res3 = layer3 + self.downsamplex3(x) + self.downsample13(res1) + self.downsample23(res2)
        ######
        layer4 = self.conv4(res3)
        layer4 = self.bn4(layer4)
        layer4 = self.relu(layer4)
        
        #layer4 = layer4 + self.downsamplex4(x)
        #layer4 = layer4 + self.downsample14(layer1)
        #layer4 = layer4 + self.downsample24(layer2)
        #layer4 = layer4 + self.downsample34(layer3)

        res4 = layer4 + self.downsamplex4(x) + self.downsample14(res1) + self.downsample24(res2) + self.downsample34(res3)
        #############
        layer5 = self.conv5(res4)
        layer5 = self.bn5(layer5)
        layer5 = self.relu(layer5)
        
        #layer5 = layer5 + self.downsamplex5(x)
        #layer5 = layer5 + self.downsample15(layer1)
        #layer5 = layer5 + self.downsample25(layer2)
        #layer5 = layer5 + self.downsample35(layer3)
        #layer5 = layer5 + self.downsample45(layer4)

        res5 = layer5 + self.downsamplex5(x) + self.downsample15(res1) + self.downsample25(res2) + self.downsample35(res3) + self.downsample45(res4)
        #######
        out = self.maxpool(layer5)
        out = flatten(out)
        out = self.fc(out)

        return out
