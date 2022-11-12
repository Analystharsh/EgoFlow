# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:24:59 2022

@author: Haowen Hu
"""

from ComputeFlow_func import main_func_even

source_path = './data/video_imgs'
val_file = './data/split/val.list'
json_path = './data/json_original'
gt_path = './data/result_LAM'
save_path = './data/optical_flow/train_val'
val_stride = 1

main_func_even(source_path, val_file, json_path, gt_path, save_path, stride=val_stride)