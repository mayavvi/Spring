# -*- coding: utf-8 -*-
# @Time : 2021/12/15 10:54 AM 
# @Author : echo
# @File : nn_module.py 
# @Software: PyCharm
import torch
from torch import nn


class Ynet(nn.Module):
    def __init__(self):
        super(Ynet, self).__init__()

    def forward(self, input):
        output = input + 1
        return output


net = Ynet()
x = torch.tensor(1.0)
output = net(x)
print(output)