# -*- coding: utf-8 -*-
# @Time : 2022/1/13 22:02 
# @Author : echo
# @File : model_pretrained.py 
# @Software: PyCharm
import torchvision

# train_data = torchvision.datasets.ImageNet("../dataset/image_net", split="train", download=True,
#                                            transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
print(vgg16_false)

