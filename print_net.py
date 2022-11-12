# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:26:34 2022

@author: Haowen Hu
"""

from model.model import BaselineLSTM, GazeLSTM
from common.config import argparser

args = argparser.parse_args()

net = eval(args.model)(args)
# net = model.model.GazeLSTM(args).cuda()

# print(net)

# ResNet
print(net.base_model.conv1.weight[0][0][0][0])
print(net.base_model_of.conv1.weight[0][0][0][0])

# for param in net.model_FlowFormer.parameters():
#     print(param)