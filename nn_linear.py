# -*- coding: utf-8 -*-
# @Time : 2022/1/9 15:02 
# @Author : echo
# @File : nn_linear.py 
# @Software: PyCharm
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)

class m(nn.Module):
    def __init__(self):
        super(m, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

net = m()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # torch.flatten()
    output = torch.reshape(imgs, (1,1,1,-1))
    print(output.shape)
    output = net(output)
    print(output.shape)