# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:10:03 2022

@author: Haowen Hu
"""

import os, cv2
import torch
import numpy as np
import time
import datetime

from configs.submission import get_cfg
from core.FlowFormer import build_flowformer

def make_test_dataset(test_path, stride=1):
    g = os.walk(test_path)
    images = []
    keyframes = []
    count = 0

    for path, dir_list, file_list in g:
        for dir_name in dir_list:
            if os.path.exists(os.path.join(test_path, dir_name)):
                uid = dir_name
                g2 = os.walk(os.path.join(test_path, uid))
                for _, track_list, _ in g2:
                    for track_id in track_list:
                        g3 = os.walk(os.path.join(test_path, uid, track_id))
                        for _, _, frame_list in g3:
                            for idx, frames in enumerate(frame_list):
                                frame_id = frames.split('_')[0]
                                unique_id = frames.split('_')[1].split('.')[0]
                                images.append((uid, track_id, unique_id, frame_id))
                                if idx % stride == 0:
                                    keyframes.append(count)
                                count += 1
    return images, keyframes

def build_model():
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))
    print(cfg.model + " loaded")

    model.cuda()
    model.eval()

    return model

def compute_flow(model, image1, image2):
    # print(image1.shape)  torch.Size([3, 280, 280])
    image1, image2 = image1[None], image2[None]  
    # print(image1.shape)  torch.Size([1, 3, 280, 280])

    # flow_pre, _ = model_FlowFormer(image1_tile, image2_tile)
    flow_pre = model(image1, image2)
    # print(len(flow_pre))  32
    # print(flow_pre[0].shape)  torch.Size([1, 2, 224, 224])
    
    return flow_pre[0]

# 保存路径形式：optical_flow/val/uid/trackid/frameid
# 有问题的光流图不存，训练、测试时读取不到的话直接设全0数组
# 计算奇数张
def main_func_odd(test_path, save_path, stride=1):
    start = time.time()
    images, keyframes = make_test_dataset(test_path, stride=stride)
    
    model = build_model().cuda()
    
    index_last = 0
    
    for index in range(len(keyframes)):
        uid, trackid, uniqueid, frameid = images[keyframes[index]]
        frameid = int(frameid)
        
        # 跳过偶数帧
        if frameid % 2 == 0:
            continue
        
        save_root = f'{save_path}/{uid}/{trackid}'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        i_1 = frameid
        i_2 = frameid + 1
        save = f'{save_path}/{uid}/{trackid}/img_{i_1:05d}.bin'
        
        if (index == 0) or (index == 1):
            start_temp = time.time()
            
        # optical flow已存在的话则跳过
        if os.path.exists(save):
            continue

        path = os.path.join(test_path, uid, trackid)
        img_1_frame = str(i_1).zfill(5)
        img_2_frame = str(i_2).zfill(5)
        g = os.walk(path)
        found_1 = False
        found_2 = False
        for _, _, file_list in g:
            for f in file_list:
                if not found_1:
                    if img_1_frame in f:
                        img_1_root = os.path.join(path, f)
                        found_1 = True
                if not found_2:
                    if img_2_frame in f:
                        img_2_root = os.path.join(path, f)
                        found_2 = True
        
        # print(img_group[uid][trackid])
        # 如果两张原图及注释都存在
        if found_1 and found_2:            
            # 读取原图，颜色通道要变换
            img_1 = cv2.imread(img_1_root)
            img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
            img_2 = cv2.imread(img_2_root)
            img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
            
            # 把image由numpy数组转为torch的tensor
            img_1 = torch.from_numpy(img_1).permute(2, 0, 1).float().cuda()
            img_2 = torch.from_numpy(img_2).permute(2, 0, 1).float().cuda()
            flow = compute_flow(model, img_1, img_2)
            # print(flow.shape)  torch.Size([1, 2, 224, 224])
            # 把flow由torch的tensor转为numpy数组
            flow = flow[0].cpu().detach().numpy()
            flow = flow.astype(np.float16)  # 测试集optical flow降低精度
            flow.tofile(save)
            # print("save_1: " + save_1 + ' saved')
            # print(flow.shape)  (2, 224, 224)
            # print(flow.dtype)  float32
        else:
            print('RGBs not exist')
                
        # 已用时间与预计时间
        if (index % 1000 == 0) and (index != index_last):
            print("----------------------------------------------------------")
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print("{} / {}".format(index, len(keyframes)))
            print("已用时间：%.3f h" % ((time.time() - start) / 3600))
            print("预计需要时间：%.3f h" % ((time.time() - start_temp) / 
                                     (index - index_last) * 
                                     (len(keyframes) - index - 1) / 3600))
            index_last = index
            start_temp = time.time()
          
# 计算偶数张 
def main_func_even(test_path, save_path, stride=1):
    start = time.time()
    images, keyframes = make_test_dataset(test_path, stride=stride)
    
    model = build_model().cuda()
    
    index_last = 0
    
    for index in range(len(keyframes)):
        uid, trackid, uniqueid, frameid = images[keyframes[index]]
        frameid = int(frameid)
        
        # 跳过奇数帧
        if frameid % 2 != 0:
            continue
        
        save_root = f'{save_path}/{uid}/{trackid}'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        i_1 = frameid
        i_2 = frameid + 1
        save = f'{save_path}/{uid}/{trackid}/img_{i_1:05d}.bin'
        
        if (index == 0) or (index == 1):
            start_temp = time.time()
            
        # optical flow已存在的话则跳过
        if os.path.exists(save):
            continue

        path = os.path.join(test_path, uid, trackid)
        img_1_frame = str(i_1).zfill(5)
        img_2_frame = str(i_2).zfill(5)
        g = os.walk(path)
        found_1 = False
        found_2 = False
        for _, _, file_list in g:
            for f in file_list:
                if not found_1:
                    if img_1_frame in f:
                        img_1_root = os.path.join(path, f)
                        found_1 = True
                if not found_2:
                    if img_2_frame in f:
                        img_2_root = os.path.join(path, f)
                        found_2 = True
        
        # print(img_group[uid][trackid])
        # 如果两张原图及注释都存在
        if found_1 and found_2:            
            # 读取原图，颜色通道要变换
            img_1 = cv2.imread(img_1_root)
            img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
            img_2 = cv2.imread(img_2_root)
            img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
            
            # 把image由numpy数组转为torch的tensor
            img_1 = torch.from_numpy(img_1).permute(2, 0, 1).float().cuda()
            img_2 = torch.from_numpy(img_2).permute(2, 0, 1).float().cuda()
            flow = compute_flow(model, img_1, img_2)
            # print(flow.shape)  torch.Size([1, 2, 224, 224])
            # 把flow由torch的tensor转为numpy数组
            flow = flow[0].cpu().detach().numpy()
            flow = flow.astype(np.float16)  # 测试集optical flow降低精度
            flow.tofile(save)
            # print("save_1: " + save_1 + ' saved')
            # print(flow.shape)  (2, 224, 224)
            # print(flow.dtype)  float32
        else:
            print('RGBs not exist')
                
        # 已用时间与预计时间
        if (index % 1000 == 0) and (index != index_last):
            print("----------------------------------------------------------")
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print("{} / {}".format(index, len(keyframes)))
            print("已用时间：%.3f h" % ((time.time() - start) / 3600))
            print("预计需要时间：%.3f h" % ((time.time() - start_temp) / 
                                     (index - index_last) * 
                                     (len(keyframes) - index - 1) / 3600))
            index_last = index
            start_temp = time.time()