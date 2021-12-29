# -*- coding: utf-8 -*-
# @Time : 2021/12/15 3:29 PM 
# @Author : echo
# @File : nn_conv2d.py 
# @Software: PyCharm

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset",train=False,
                                       transform=torchvision.transforms.Tensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Ynet(nn.Moudle):
    def __init__(self):
        super(Ynet, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


ynet = Ynet()
print(ynet)
