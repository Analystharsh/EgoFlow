# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 16:38:55 2022

@author: Haowen Hu
"""

import torch, math
import torch.optim
import torch.utils.data
import copy

from configs.submission import get_cfg
from core.FlowFormer import build_flowformer
import torch.nn.functional as F

TRAIN_SIZE = [224, 224]

def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
  if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  if image_shape[0] == TRAIN_SIZE[0]:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
  else:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
  if image_shape[1] == TRAIN_SIZE[1]:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
  else:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

  # Make sure the final patch is flush with the image boundary
  hs[-1] = image_shape[0] - patch_size[0]
  ws[-1] = image_shape[1] - patch_size[1]
  return [(h, w) for h in hs for w in ws]

def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5 
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.to(1)
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights
    
def build_FlowFormer():
    cfg = get_cfg()
    model_FlowFormer = torch.nn.DataParallel(build_flowformer(cfg))
    model_FlowFormer.load_state_dict(torch.load(cfg.model))

    model_FlowFormer.to(1)
    model_FlowFormer.eval()

    return model_FlowFormer

def compute_flow(model_FlowFormer, image1, image2, weights=None):
    image_size = image1.shape[1:]
    
    # print(image_size)  torch.Size([224, 224])
    hws = compute_grid_indices(image_size)    
    # print(hws)  [(0, 0)]  如果image长宽都是224的话，就会输出[(0, 0)]
    if weights is None:
        weights = compute_weight(hws, image_size, sigma=0.05)
    # print(len(weights))  1
    # print(weights[0].shape)  torch.Size([1, 1, 224, 224])

    # image1, image2 = image1[None].cuda(), image2[None].cuda()

    flows = 0
    flow_count = 0

    for idx, (h, w) in enumerate(hws):
        image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
        image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]  
        # print(image1_tile.shape)  torch.Size([1, 3, 224, 224])
        flow_pre, _ = model_FlowFormer(image1_tile, image2_tile)
        padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
        # print(len(flow_pre))  32
        # print(flow_pre[0].shape)  torch.Size([1, 2, 224, 224])
        # print(padding)  (0, 0, 0, 0, 0, 0)
        flows += F.pad(flow_pre * weights[idx], padding)
        print(flows.shape)
        # flow_count += F.pad(weights, padding)
        flow_count += F.pad(weights[idx], padding)

    flow_pre = flows / flow_count
    # flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

    # return flow, weights
    return flow_pre, weights
    
# 用RGB生成Optical Flow
def get_video_of(source_video, FlowFormer_model):
    print("-----------------------source video of----------------------------")
    # print(len(source_video))  7
    # print(source_video[0].shape)  torch.Size([3, 224, 224])
    source_video_cuda = copy.deepcopy(source_video).to(1)
    weights = None
    source_video_of = []
    for img1_id in range(len(source_video_cuda) - 1):
        img1 = source_video_cuda[img1_id]
        img2 = source_video_cuda[img1_id + 1]
        # 先扩充为torch.Size([3, 224, 280])
        zeros = torch.zeros(3, 224, 56).to(1)
        img1 = torch.cat([img1, zeros], 2)
        img2 = torch.cat([img2, zeros], 2)
        # print(img1.shape)  torch.Size([3, 224, 280])
        # print(img1.dtype)  torch.float32
        flow, weights = compute_flow(FlowFormer_model, img1, img2, weights)
        print(flow.shape)
        source_video_of.append(flow)
    source_video_of = torch.tensor(source_video_of)
    print(source_video_of.shape)
    source_video_of.cpu()
    
    return source_video_of