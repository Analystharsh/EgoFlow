# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:10:03 2022

@author: Haowen Hu
"""

import os, cv2, json, glob, logging
import torch
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict, OrderedDict
import time
import datetime

from configs.submission import get_cfg
from core.FlowFormer import build_flowformer

def helper():
    return defaultdict(OrderedDict)

def _nested_dict():
    return defaultdict(helper)

def _get_img_group(images):
    img_group = _nested_dict()
    for db in images:
        img_group[db[0]][db[1]][db[2]] = db[3]
    return img_group

def check(track):
    inter_track = []
    framenum = []
    bboxes = []
    for frame in track:
        x = frame['x']
        y = frame['y']
        w = frame['width']
        h = frame['height']
        if (w <= 0 or h <= 0 or
                frame['frameNumber'] == 0 or
                len(frame['Person ID']) == 0):
            continue
        framenum.append(frame['frameNumber'])
        x = max(x, 0)
        y = max(y, 0)
        bbox = [x, y, x + w, y + h]
        bboxes.append(bbox)

    if len(framenum) == 0:
        return inter_track

    framenum = np.array(framenum)
    bboxes = np.array(bboxes)

    gt_frames = framenum[-1] - framenum[0] + 1

    frame_i = np.arange(framenum[0], framenum[-1] + 1)

    if gt_frames > framenum.shape[0]:
        bboxes_i = []
        for ij in range(0, 4):
            interpfn = interp1d(framenum, bboxes[:, ij])
            bboxes_i.append(interpfn(frame_i))
        bboxes_i = np.stack(bboxes_i, axis=1)
    else:
        frame_i = framenum
        bboxes_i = bboxes

    # assemble new tracklet
    template = track[0]
    for i, (frame, bbox) in enumerate(zip(frame_i, bboxes_i)):
        record = template.copy()
        record['frameNumber'] = frame
        record['x'] = bbox[0]
        record['y'] = bbox[1]
        record['width'] = bbox[2] - bbox[0]
        record['height'] = bbox[3] - bbox[1]
        inter_track.append(record)
    return inter_track

def make_dataset(file_name, json_path, gt_path, stride=1):
    images = []
    keyframes = []
    count = 0

    with open(file_name, 'r') as f:
        videos = f.readlines()
    for uid in videos:
        uid = uid.strip()

        with open(os.path.join(gt_path, uid + '.json')) as f:
            gts = json.load(f)
        positive = set()
        for gt in gts:
            for i in range(gt['start_frame'], gt['end_frame'] + 1):
                positive.add(str(i) + ":" + gt['label'])

        vid_json_dir = os.path.join(json_path, uid)
        tracklets = glob.glob(f'{vid_json_dir}/*.json')
        for t in tracklets:
            with open(t, 'r') as j:
                frames = json.load(j)
            frames.sort(key=lambda x: x['frameNumber'])
            trackid = os.path.basename(t)[:-5]
            # check the bbox, interpolate when necessary
            frames = check(frames)

            for idx, frame in enumerate(frames):
                frameid = frame['frameNumber']
                bbox = (frame['x'],
                        frame['y'],
                        frame['x'] + frame['width'],
                        frame['y'] + frame['height'])
                identifier = str(frameid) + ':' + frame['Person ID']
                label = 1 if identifier in positive else 0
                images.append((uid, trackid, frameid, bbox, label))
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
# 保存奇数张
def main_func_odd(source_path, file_name, json_path, gt_path, save_path,
                  stride=1, scale=0):
    start = time.time()
    images, keyframes = make_dataset(file_name, json_path, gt_path, stride=stride)
    img_group = _get_img_group(images)
    # print(img_group)
    
    model = build_model().cuda()
    
    index_last = 0
    
    for index in range(len(keyframes)):
        uid, trackid, frameid, _, label = images[keyframes[index]]
        # print(frameid)
        
        # 跳过偶数帧
        if frameid % 2 == 0:
            continue
        
        save_root = f'{save_path}/{uid}/{trackid}'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        i_1 = frameid
        i_2 = frameid + 1
        save = f'{save_path}/{uid}/{trackid}/img_{i_1:05d}.bin'
        
        # 两张原图的路径
        # print(i_1)
        # print(i_2)
        # print(i_3)
        img_1_root = f'{source_path}/{uid}/img_{i_1:05d}.jpg'
        img_2_root = f'{source_path}/{uid}/img_{i_2:05d}.jpg'
        
        if (index == 0) or (index == 1):
            start_temp = time.time()
        
        if not os.path.exists(save):
            # print(img_group[uid][trackid])
            # 如果两张原图及注释都存在
            if (i_1 in img_group[uid][trackid] and os.path.exists(img_1_root)) and (
                i_2 in img_group[uid][trackid] and os.path.exists(img_2_root)):            
                # 读取原图，颜色通道要变换
                img_1 = cv2.imread(img_1_root)
                img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
                img_2 = cv2.imread(img_2_root)
                img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
            
                # 截取face
                bbox = img_group[uid][trackid][i_1]
                x1 = int((1.0 - scale) * bbox[0])
                y1 = int((1.0 - scale) * bbox[1])
                x2 = int((1.0 + scale) * bbox[2])
                y2 = int((1.0 + scale) * bbox[3])
                face_1 = img_1[y1: y2, x1: x2, :]
                bbox = img_group[uid][trackid][i_2]
                x1 = int((1.0 - scale) * bbox[0])
                y1 = int((1.0 - scale) * bbox[1])
                x2 = int((1.0 + scale) * bbox[2])
                y2 = int((1.0 + scale) * bbox[3])
                face_2 = img_2[y1: y2, x1: x2, :]
                if_cal = True
                try:
                    face_1 = cv2.resize(face_1, (224, 224))
                    face_2 = cv2.resize(face_2, (224, 224))
                except:
                    # bad bbox
                    print('bad bbox when calculating optical flow')
                    if_cal = False
                
                # 若两张face都正常，则计算optical flow
                if if_cal:
                    # 把face由numpy数组转为torch的tensor
                    face_1 = torch.from_numpy(face_1).permute(2, 0, 1).float().cuda()
                    face_2 = torch.from_numpy(face_2).permute(2, 0, 1).float().cuda()
                    flow = compute_flow(model, face_1, face_2)
                    # print(flow.shape)  torch.Size([1, 2, 224, 224])
                    # 把flow由torch的tensor转为numpy数组
                    flow = flow[0].cpu().detach().numpy()
                    flow = flow.astype(np.float16)  # 测试集optical flow降低精度
                    flow.tofile(save)
                    # print("save_1: " + save_1 + ' saved')
                    # print(flow.shape)  (2, 224, 224)
                    # print(flow.dtype)  float32
            else:
                print("RGBs or annotations for save_1 not exist")
                
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
            
# 保存偶数张
def main_func_even(source_path, file_name, json_path, gt_path, save_path,
                   stride=1, scale=0):
    start = time.time()
    images, keyframes = make_dataset(file_name, json_path, gt_path, stride=stride)
    img_group = _get_img_group(images)
    # print(img_group)
    
    model = build_model().cuda()
    
    index_last = 0
    
    for index in range(len(keyframes)):
        uid, trackid, frameid, _, label = images[keyframes[index]]
        # print(frameid)
        
        # 跳过奇数帧
        if frameid % 2 != 0:
            continue
        
        save_root = f'{save_path}/{uid}/{trackid}'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        i_1 = frameid
        i_2 = frameid + 1
        save = f'{save_path}/{uid}/{trackid}/img_{i_1:05d}.bin'
        
        # 两张原图的路径
        # print(i_1)
        # print(i_2)
        # print(i_3)
        img_1_root = f'{source_path}/{uid}/img_{i_1:05d}.jpg'
        img_2_root = f'{source_path}/{uid}/img_{i_2:05d}.jpg'
        
        if (index == 0) or (index == 1):
            start_temp = time.time()
        
        if not os.path.exists(save):
            # print(img_group[uid][trackid])
            # 如果两张原图及注释都存在
            if (i_1 in img_group[uid][trackid] and os.path.exists(img_1_root)) and (
                i_2 in img_group[uid][trackid] and os.path.exists(img_2_root)):            
                # 读取原图，颜色通道要变换
                img_1 = cv2.imread(img_1_root)
                img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
                img_2 = cv2.imread(img_2_root)
                img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
            
                # 截取face
                bbox = img_group[uid][trackid][i_1]
                x1 = int((1.0 - scale) * bbox[0])
                y1 = int((1.0 - scale) * bbox[1])
                x2 = int((1.0 + scale) * bbox[2])
                y2 = int((1.0 + scale) * bbox[3])
                face_1 = img_1[y1: y2, x1: x2, :]
                bbox = img_group[uid][trackid][i_2]
                x1 = int((1.0 - scale) * bbox[0])
                y1 = int((1.0 - scale) * bbox[1])
                x2 = int((1.0 + scale) * bbox[2])
                y2 = int((1.0 + scale) * bbox[3])
                face_2 = img_2[y1: y2, x1: x2, :]
                if_cal = True
                try:
                    face_1 = cv2.resize(face_1, (224, 224))
                    face_2 = cv2.resize(face_2, (224, 224))
                except:
                    # bad bbox
                    print('bad bbox when calculating optical flow')
                    if_cal = False
                
                # 若两张face都正常，则计算optical flow
                if if_cal:
                    # 把face由numpy数组转为torch的tensor
                    face_1 = torch.from_numpy(face_1).permute(2, 0, 1).float().cuda()
                    face_2 = torch.from_numpy(face_2).permute(2, 0, 1).float().cuda()
                    flow = compute_flow(model, face_1, face_2)
                    # print(flow.shape)  torch.Size([1, 2, 224, 224])
                    # 把flow由torch的tensor转为numpy数组
                    flow = flow[0].cpu().detach().numpy()
                    flow = flow.astype(np.float16)  # 测试集optical flow降低精度
                    flow.tofile(save)
                    # print("save_1: " + save_1 + ' saved')
                    # print(flow.shape)  (2, 224, 224)
                    # print(flow.dtype)  float32
            else:
                print("RGBs or annotations for save_1 not exist")
                
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

def main_func(source_path, file_name, json_path, gt_path, save_path,
              stride=1, scale=0):
    start = time.time()
    images, keyframes = make_dataset(file_name, json_path, gt_path, stride=stride)
    img_group = _get_img_group(images)
    # print(img_group)
    
    model = build_model().cuda()
    
    for index in range(len(keyframes)):
        uid, trackid, frameid, _, label = images[keyframes[index]]
        # print(frameid)
        
        save_root = f'{save_path}/{uid}/{trackid}'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        i_1 = frameid - 1
        i_2 = frameid
        save_1 = f'{save_path}/{uid}/{trackid}/img_{i_1:05d}.bin'
        save_2 = f'{save_path}/{uid}/{trackid}/img_{i_2:05d}.bin'
        
        # 三张原图的路径
        i_3 = frameid + 1
        # print(i_1)
        # print(i_2)
        # print(i_3)
        img_1_root = f'{source_path}/{uid}/img_{i_1:05d}.jpg'
        img_2_root = f'{source_path}/{uid}/img_{i_2:05d}.jpg'
        img_3_root = f'{source_path}/{uid}/img_{i_3:05d}.jpg'
        
        if index == 0:
            start_temp = time.time()
        
        if not os.path.exists(save_1):
            # print(img_group[uid][trackid])
            # 如果前两张原图及注释都存在
            if (i_1 in img_group[uid][trackid] and os.path.exists(img_1_root)) and (
                i_2 in img_group[uid][trackid] and os.path.exists(img_2_root)):            
                # 读取原图，颜色通道要变换
                img_1 = cv2.imread(img_1_root)
                img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
                img_2 = cv2.imread(img_2_root)
                img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
            
                # 截取face
                bbox = img_group[uid][trackid][i_1]
                x1 = int((1.0 - scale) * bbox[0])
                y1 = int((1.0 - scale) * bbox[1])
                x2 = int((1.0 + scale) * bbox[2])
                y2 = int((1.0 + scale) * bbox[3])
                face_1 = img_1[y1: y2, x1: x2, :]
                bbox = img_group[uid][trackid][i_2]
                x1 = int((1.0 - scale) * bbox[0])
                y1 = int((1.0 - scale) * bbox[1])
                x2 = int((1.0 + scale) * bbox[2])
                y2 = int((1.0 + scale) * bbox[3])
                face_2 = img_2[y1: y2, x1: x2, :]
                if_cal = True
                try:
                    face_1 = cv2.resize(face_1, (224, 224))
                    face_2 = cv2.resize(face_2, (224, 224))
                except:
                    # bad bbox
                    print('bad bbox when calculating optical flow')
                    if_cal = False
                
                # 若两张face都正常，则计算optical flow
                if if_cal:
                    # 把face由numpy数组转为torch的tensor
                    face_1 = torch.from_numpy(face_1).permute(2, 0, 1).float().cuda()
                    face_2 = torch.from_numpy(face_2).permute(2, 0, 1).float().cuda()
                    flow = compute_flow(model, face_1, face_2)
                    # print(flow.shape)  torch.Size([1, 2, 224, 224])
                    # 把flow由torch的tensor转为numpy数组
                    flow = flow[0].cpu().detach().numpy()
                    flow.tofile(save_1)
                    # print("save_1: " + save_1 + ' saved')
                    # print(flow.shape)  (2, 224, 224)
                    # print(flow.dtype)  float32
            else:
                print("RGBs or annotations for save_1 not exist")
                
        if not os.path.exists(save_2):
            # 如果后两张原图及注释都存在
            if (i_2 in img_group[uid][trackid] and os.path.exists(img_2_root)) and (
                i_3 in img_group[uid][trackid] and os.path.exists(img_3_root)):
                # 读取原图，颜色通道要变换
                img_2 = cv2.imread(img_2_root)
                img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
                img_3 = cv2.imread(img_3_root)
                img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB)
            
                # 截取face
                bbox = img_group[uid][trackid][i_2]
                x1 = int((1.0 - scale) * bbox[0])
                y1 = int((1.0 - scale) * bbox[1])
                x2 = int((1.0 + scale) * bbox[2])
                y2 = int((1.0 + scale) * bbox[3])
                face_2 = img_2[y1: y2, x1: x2, :]
                bbox = img_group[uid][trackid][i_3]
                x1 = int((1.0 - scale) * bbox[0])
                y1 = int((1.0 - scale) * bbox[1])
                x2 = int((1.0 + scale) * bbox[2])
                y2 = int((1.0 + scale) * bbox[3])
                face_3 = img_3[y1: y2, x1: x2, :]
                if_cal = True
                try:
                    face_2 = cv2.resize(face_2, (224, 224))
                    face_3 = cv2.resize(face_3, (224, 224))
                except:
                    # bad bbox
                    print('bad bbox when calculating optical flow')
                    if_cal = False
                
                # 若两张face都正常，则计算optical flow
                if if_cal:
                    # 把face由numpy数组转为torch的tensor
                    face_2 = torch.from_numpy(face_2).permute(2, 0, 1).float().cuda()
                    face_3 = torch.from_numpy(face_3).permute(2, 0, 1).float().cuda()
                    flow = compute_flow(model, face_2, face_3)
                    # print(flow.shape)  torch.Size([1, 2, 224, 224])
                    # 把flow由torch的tensor转为numpy数组
                    flow = flow[0].cpu().detach().numpy()
                    flow.tofile(save_2)
                    # print("save_2: " + save_2 + ' saved')
            else:
                print("RGBs or annotations for save_2 not exist")
                
        # 已用时间与预计时间
        if index % 1000 == 0:
            print("----------------------------------------------------------")
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print("{} / {}".format(index, len(keyframes)))
            print("已用时间：%.3f h" % ((time.time() - start) / 3600))
            print("预计需要时间：%.3f h" % ((time.time() - start_temp) / 1000 * 
                                     (len(keyframes) - index - 1) / 3600))
            start_temp = time.time()
        
def main_func_first2(source_path, file_name, json_path, gt_path, save_path,
                     stride=1, scale=0):
    start = time.time()
    images, keyframes = make_dataset(file_name, json_path, gt_path, stride=stride)
    img_group = _get_img_group(images)
    # print(img_group)
    
    model = build_model().cuda()
    
    for index in range(len(keyframes)):
        uid, trackid, frameid, _, label = images[keyframes[index]]
        # print(frameid)
        
        save_root = f'{save_path}/{uid}/{trackid}'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        i_1 = frameid - 3
        i_2 = frameid - 2
        save_1 = f'{save_path}/{uid}/{trackid}/img_{i_1:05d}.bin'
        save_2 = f'{save_path}/{uid}/{trackid}/img_{i_2:05d}.bin'
        
        # 三张原图的路径
        i_3 = frameid - 1
        # print(i_1)
        # print(i_2)
        # print(i_3)
        img_1_root = f'{source_path}/{uid}/img_{i_1:05d}.jpg'
        img_2_root = f'{source_path}/{uid}/img_{i_2:05d}.jpg'
        img_3_root = f'{source_path}/{uid}/img_{i_3:05d}.jpg'
        
        if index == 0:
            start_temp = time.time()
        
        if not os.path.exists(save_1):
            # print(img_group[uid][trackid])
            # 如果前两张原图及注释都存在
            if (i_1 in img_group[uid][trackid] and os.path.exists(img_1_root)) and (
                i_2 in img_group[uid][trackid] and os.path.exists(img_2_root)):
                # 读取原图，颜色通道要变换
                img_1 = cv2.imread(img_1_root)
                img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
                img_2 = cv2.imread(img_2_root)
                img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
            
                # 截取face
                bbox = img_group[uid][trackid][i_1]
                x1 = int((1.0 - scale) * bbox[0])
                y1 = int((1.0 - scale) * bbox[1])
                x2 = int((1.0 + scale) * bbox[2])
                y2 = int((1.0 + scale) * bbox[3])
                face_1 = img_1[y1: y2, x1: x2, :]
                bbox = img_group[uid][trackid][i_2]
                x1 = int((1.0 - scale) * bbox[0])
                y1 = int((1.0 - scale) * bbox[1])
                x2 = int((1.0 + scale) * bbox[2])
                y2 = int((1.0 + scale) * bbox[3])
                face_2 = img_2[y1: y2, x1: x2, :]
                if_cal = True
                try:
                    face_1 = cv2.resize(face_1, (224, 224))
                    face_2 = cv2.resize(face_2, (224, 224))
                except:
                    # bad bbox
                    print('bad bbox when calculating optical flow')
                    if_cal = False
                
                # 若两张face都正常，则计算optical flow
                if if_cal:
                    # 把face由numpy数组转为torch的tensor
                    face_1 = torch.from_numpy(face_1).permute(2, 0, 1).float().cuda()
                    face_2 = torch.from_numpy(face_2).permute(2, 0, 1).float().cuda()
                    flow = compute_flow(model, face_1, face_2)
                    # print(flow.shape)  torch.Size([1, 2, 224, 224])
                    # 把flow由torch的tensor转为numpy数组
                    flow = flow[0].cpu().detach().numpy()
                    flow.tofile(save_1)
                    # print("save_1: " + save_1 + ' saved')
                    # print(flow.shape)  (2, 224, 224)
                    # print(flow.dtype)  float32
            else:
                print("RGBs or annotations for save_1 not exist")
             
        if not os.path.exists(save_2):   
            # 如果后两张原图及注释都存在
            if (i_2 in img_group[uid][trackid] and os.path.exists(img_2_root)) and (
                i_3 in img_group[uid][trackid] and os.path.exists(img_3_root)):
                # 读取原图，颜色通道要变换
                img_2 = cv2.imread(img_2_root)
                img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
                img_3 = cv2.imread(img_3_root)
                img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB)
            
                # 截取face
                bbox = img_group[uid][trackid][i_2]
                x1 = int((1.0 - scale) * bbox[0])
                y1 = int((1.0 - scale) * bbox[1])
                x2 = int((1.0 + scale) * bbox[2])
                y2 = int((1.0 + scale) * bbox[3])
                face_2 = img_2[y1: y2, x1: x2, :]
                bbox = img_group[uid][trackid][i_3]
                x1 = int((1.0 - scale) * bbox[0])
                y1 = int((1.0 - scale) * bbox[1])
                x2 = int((1.0 + scale) * bbox[2])
                y2 = int((1.0 + scale) * bbox[3])
                face_3 = img_3[y1: y2, x1: x2, :]
                if_cal = True
                try:
                    face_2 = cv2.resize(face_2, (224, 224))
                    face_3 = cv2.resize(face_3, (224, 224))
                except:
                    # bad bbox
                    print('bad bbox when calculating optical flow')
                    if_cal = False
                
                # 若两张face都正常，则计算optical flow
                if if_cal:
                    # 把face由numpy数组转为torch的tensor
                    face_2 = torch.from_numpy(face_2).permute(2, 0, 1).float().cuda()
                    face_3 = torch.from_numpy(face_3).permute(2, 0, 1).float().cuda()
                    flow = compute_flow(model, face_2, face_3)
                    # print(flow.shape)  torch.Size([1, 2, 224, 224])
                    # 把flow由torch的tensor转为numpy数组
                    flow = flow[0].cpu().detach().numpy()
                    flow.tofile(save_2)
                    # print("save_2: " + save_2 + ' saved')
            else:
                print("RGBs or annotations for save_2 not exist")
                
        # 已用时间与预计时间
        if index % 1000 == 0:
            print("----------------------------------------------------------")
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print("{} / {}".format(index, len(keyframes)))
            print("已用时间：%.3f h" % ((time.time() - start) / 3600))
            print("预计需要时间：%.3f h" % ((time.time() - start_temp) / 1000 * 
                                     (len(keyframes) - index - 1) / 3600))
            start_temp = time.time()
        
def main_func_last2(source_path, file_name, json_path, gt_path, save_path,
                     stride=1, scale=0):
    start = time.time()
    images, keyframes = make_dataset(file_name, json_path, gt_path, stride=stride)
    img_group = _get_img_group(images)
    # print(img_group)
    
    model = build_model().cuda()
    
    for index in range(len(keyframes)):
        uid, trackid, frameid, _, label = images[keyframes[index]]
        # print(frameid)
        
        save_root = f'{save_path}/{uid}/{trackid}'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        i_1 = frameid + 1
        i_2 = frameid + 2
        save_1 = f'{save_path}/{uid}/{trackid}/img_{i_1:05d}.bin'
        save_2 = f'{save_path}/{uid}/{trackid}/img_{i_2:05d}.bin'
        
        # 三张原图的路径
        i_3 = frameid + 3
        # print(i_1)
        # print(i_2)
        # print(i_3)
        img_1_root = f'{source_path}/{uid}/img_{i_1:05d}.jpg'
        img_2_root = f'{source_path}/{uid}/img_{i_2:05d}.jpg'
        img_3_root = f'{source_path}/{uid}/img_{i_3:05d}.jpg'
        
        if index == 0:
            start_temp = time.time()
        
        if not os.path.exists(save_1):
            # print(img_group[uid][trackid])
            # 如果前两张原图及注释都存在
            if (i_1 in img_group[uid][trackid] and os.path.exists(img_1_root)) and (
                i_2 in img_group[uid][trackid] and os.path.exists(img_2_root)):
                # 读取原图，颜色通道要变换
                img_1 = cv2.imread(img_1_root)
                img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
                img_2 = cv2.imread(img_2_root)
                img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
            
                # 截取face
                bbox = img_group[uid][trackid][i_1]
                x1 = int((1.0 - scale) * bbox[0])
                y1 = int((1.0 - scale) * bbox[1])
                x2 = int((1.0 + scale) * bbox[2])
                y2 = int((1.0 + scale) * bbox[3])
                face_1 = img_1[y1: y2, x1: x2, :]
                bbox = img_group[uid][trackid][i_2]
                x1 = int((1.0 - scale) * bbox[0])
                y1 = int((1.0 - scale) * bbox[1])
                x2 = int((1.0 + scale) * bbox[2])
                y2 = int((1.0 + scale) * bbox[3])
                face_2 = img_2[y1: y2, x1: x2, :]
                if_cal = True
                try:
                    face_1 = cv2.resize(face_1, (224, 224))
                    face_2 = cv2.resize(face_2, (224, 224))
                except:
                    # bad bbox
                    print('bad bbox when calculating optical flow')
                    if_cal = False
                
                # 若两张face都正常，则计算optical flow
                if if_cal:
                    # 把face由numpy数组转为torch的tensor
                    face_1 = torch.from_numpy(face_1).permute(2, 0, 1).float().cuda()
                    face_2 = torch.from_numpy(face_2).permute(2, 0, 1).float().cuda()
                    flow = compute_flow(model, face_1, face_2)
                    # print(flow.shape)  torch.Size([1, 2, 224, 224])
                    # 把flow由torch的tensor转为numpy数组
                    flow = flow[0].cpu().detach().numpy()
                    flow.tofile(save_1)
                    # print("save_1: " + save_1 + ' saved')
                    # print(flow.shape)  (2, 224, 224)
                    # print(flow.dtype)  float32
            else:
                print("RGBs or annotations for save_1 not exist")
             
        if not os.path.exists(save_2):   
            # 如果后两张原图及注释都存在
            if (i_2 in img_group[uid][trackid] and os.path.exists(img_2_root)) and (
                i_3 in img_group[uid][trackid] and os.path.exists(img_3_root)):
                # 读取原图，颜色通道要变换
                img_2 = cv2.imread(img_2_root)
                img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
                img_3 = cv2.imread(img_3_root)
                img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB)
            
                # 截取face
                bbox = img_group[uid][trackid][i_2]
                x1 = int((1.0 - scale) * bbox[0])
                y1 = int((1.0 - scale) * bbox[1])
                x2 = int((1.0 + scale) * bbox[2])
                y2 = int((1.0 + scale) * bbox[3])
                face_2 = img_2[y1: y2, x1: x2, :]
                bbox = img_group[uid][trackid][i_3]
                x1 = int((1.0 - scale) * bbox[0])
                y1 = int((1.0 - scale) * bbox[1])
                x2 = int((1.0 + scale) * bbox[2])
                y2 = int((1.0 + scale) * bbox[3])
                face_3 = img_3[y1: y2, x1: x2, :]
                if_cal = True
                try:
                    face_2 = cv2.resize(face_2, (224, 224))
                    face_3 = cv2.resize(face_3, (224, 224))
                except:
                    # bad bbox
                    print('bad bbox when calculating optical flow')
                    if_cal = False
                
                # 若两张face都正常，则计算optical flow
                if if_cal:
                    # 把face由numpy数组转为torch的tensor
                    face_2 = torch.from_numpy(face_2).permute(2, 0, 1).float().cuda()
                    face_3 = torch.from_numpy(face_3).permute(2, 0, 1).float().cuda()
                    flow = compute_flow(model, face_2, face_3)
                    # print(flow.shape)  torch.Size([1, 2, 224, 224])
                    # 把flow由torch的tensor转为numpy数组
                    flow = flow[0].cpu().detach().numpy()
                    flow.tofile(save_2)
                    # print("save_2: " + save_2 + ' saved')
            else:
                print("RGBs or annotations for save_2 not exist")
                
        # 已用时间与预计时间
        if index % 1000 == 0:
            print("----------------------------------------------------------")
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print("{} / {}".format(index, len(keyframes)))
            print("已用时间：%.3f h" % ((time.time() - start) / 3600))
            print("预计需要时间：%.3f h" % ((time.time() - start_temp) / 1000 * 
                                     (len(keyframes) - index - 1) / 3600))
            start_temp = time.time()