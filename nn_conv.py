# -*- coding: utf-8 -*-
# @Time : 2021/12/15 3:04 PM 
# @Author : echo
# @File : nn_conv.py 
# @Software: PyCharm

import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(kernel.shape)

output1 = F.conv2d(input, kernel, stride=1)
print(output1)

output2 = F.conv2d(input, kernel, stride=2)
print(output2.shape)

output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)