# -*- coding: utf-8 -*-
# @Time : 2022/1/7 22:35 
# @Author : echo
# @File : nn_maxpool.py 
# @Software: PyCharm

import torch

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])
input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)
# torch.Size([1, 1, 5, 5])