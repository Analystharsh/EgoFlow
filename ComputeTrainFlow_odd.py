# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:24:59 2022

@author: Haowen Hu
"""

from ComputeFlow_func import main_func_odd

source_path = './data/video_imgs'
train_file = './data/split/train.list'
json_path = './data/json_original'
gt_path = './data/result_LAM'
save_path = './data/optical_flow/train_val'
train_stride = 1

main_func_odd(source_path, train_file, json_path, gt_path, save_path, 
              stride=train_stride)