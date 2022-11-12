# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:24:59 2022

@author: Haowen Hu
"""

from ComputeTestFlow_func import main_func_odd

test_path = '/media/ssd1/hu/videos_challenge'
save_path = '/media/ssd1/hu/optical_flow/test'
test_stride = 1

main_func_odd(test_path, save_path, stride=test_stride)