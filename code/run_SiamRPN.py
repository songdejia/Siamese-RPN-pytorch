# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-11-05 19:29:07
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-11-08 17:07:31
# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import os
import sys


from utils import get_subwindow_tracking
"""
target size 127
stride 8
detection size 271
total 127 + (19-1)*8 = 271
"""


def generate_anchor(total_stride, scales, ratios, score_size):
    """
    生成anchor
    total_stride 8
    scales = [8, ]
    ratios = [0.33, 0.5, 1, 2, 3]
    score_size = 19

    产生top-left and w,h的
    """
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    size = total_stride * total_stride # 8 * 8
    count = 0
    #这里相当于是计算了一个位置的anchor 
    #这个位置各个尺度各个比例都算了wh ，xy 待定
    #每个位置都wh是确定的
    for ratio in ratios:
        ws = int(np.sqrt(size / ratio)) # 8 / sqrt(ratio)
        hs = int(ws * ratio)            # 8 * sqrt(ratio)
        for scale in scales:
            wws = ws * scale            # 64 / sqrt(ratio)
            hhs = hs * scale            # 64 * sqrt(ratio)
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    #重复把每个位置都anchor都堆叠起来
    #并填充中心点
    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size / 2) * total_stride #这里anchor是以中心为对称点的9，1，9
    #print([ori + total_stride * dx for dx in range(score_size)])
    #[-72, -64, -56, -48, -40, -32, -24, -16, -8, 0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
    #
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)

    return anchor


class TrackerConfig(object):
    # These are the default hyper-params for DaSiamRPN 0.3827
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # input z size
    instance_size = 271  # input x size (search region)
    total_stride  = 8
    score_size = (instance_size-exemplar_size)/total_stride+1
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    template = None


def tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p, ids, name, original_img, root_path = '/home/song/srpn/tmp'):
    delta, score = net(x_crop) 

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()
    """
    for i in range(p.anchor.shape[0]):
        print('anchor  ====>  {}'.format(p.anchor[i]))
    """
    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]
    # delta[0, :] is x = delta * w + x
    # delta[1, :] is y = delta * y + y
    # delta[2, :] is w = exp(w, delta)
    # delta[3, :] is h = exp(h, delta)

    # compute change ratio (r, 1/r)
    def change(r):
        return np.maximum(r, 1./r)

    # compute size of larger area
    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)
    # compute size of larger area, input []
    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    ##############################################################
    # score => pscore => pscore+window
    # size penalty, delta is proposal
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty, bbox scale ratio(area ratio)
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore = penalty * score

    # window float, 
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    target = delta[:, best_pscore_id] / scale_z
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr # a kind of score

    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])

    ##### vis#################################################
    im_h, im_w, _ = original_img.shape
    res_x = max(0, min(im_w, target_pos[0]))
    res_y = max(0, min(im_h, target_pos[1]))
    res_w = max(10, min(im_w, target_sz[0]))
    res_h = max(10, min(im_h, target_sz[1]))

    x1 = res_x - res_w/2
    x2 = res_x + res_w/2
    x3 = x2
    x4 = x1
    y1 = res_y - res_h/2
    y2 = y1
    y3 = res_y + res_h/2
    y4 = y3
    box = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    im = x_crop #(1L, 3L, 271L, 271L
    im = im.squeeze(0).permute((1,2,0)).data.cpu().numpy()
    cv2.polylines(original_img, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
    save_dir_path = os.path.join(root_path, name)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    img_file = os.path.join(save_dir_path, '{:03d}_detection_output.jpg'.format(ids+1))
    cv2.imwrite(img_file, original_img)
    print('save at {}'.format(img_file))
    ##################################################################
    return target_pos, target_sz, score[best_pscore_id]


def SiamRPN_init(im, target_pos, target_sz, net):
    """
    输入第一帧
    target_pos [center_x, center_y]
    target_sz  [w, h]
    net 


    return
    state['im_h']
    state['im_w']
    state['p']  config for tracker
    state['net']
    state['avg_chan'] 通道均值
    """
    state = dict()
    p = TrackerConfig()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    # Input size, if target is small, input should be large?
    if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
        p.instance_size = 287  # small object big search region
    else:
        p.instance_size = 271

    # Input size - Template size
    # 计算每行有多少个感受野
    # 每个感受野size instance_size
    # 每次移动total_stride
    p.score_size = (p.instance_size - p.exemplar_size) / p.total_stride + 1

    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, p.score_size)

    # 每在一维做平均则下降一维
    # 1024 * 1024 * 3 => [x1, x2, x3]
    avg_chans = np.mean(im, axis=(0, 1))

    # 扩大template范围
    # 并且需要归一成正方形
    # detection不需要归一
    # w_ -> w + (w+h)/2
    # h_ -> h + (w+h)/2 
    # s_ -> sqrt(w_ * h_)
    # target是实际bg 而s_z是相当于把bg变成了正方形
    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    
    # initialize the exemplar
    # 将溢出部分用avg补充
    # target_pos是中心点
    # s_z是归一后正方形大小
    # exempler_size是后面需要resize的127
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
    template = z_crop.numpy().transpose((1,2,0))
    state['template']=template

    z = Variable(z_crop.unsqueeze(0))
    net.temple(z.cuda())

    if p.windowing == 'cosine':
        #outer (x1, x2)
        #x1中的每个值变为x2行向量的倍数
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)#np.tile复制(row, col)倍 or directly copy x

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz

    return state


def SiamRPN_track(state, im, ids, name):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz'] #background bbox

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z #scale ratio of template
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    # 这里相当于目标的位置仍然在原位
    # 然后以此中心截取 s_x 并且做缩放
    # 这样做的缺点在于如果高速移动 就容易crop不到
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
    #print(x_crop.shape)#(1L, 3L, 271L, 271L)
    save_img = x_crop.data.squeeze(0).numpy().transpose((1,2,0)).astype(np.int32)
    save_path = os.path.join('/home/song/srpn/tmp', name, '{:03d}_detection_input.jpg'.format(ids))
    cv2.imwrite(save_path, save_img)
    print('save detection input image @ {}'.format(save_path))

    target_pos, target_sz, score = tracker_eval(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p, ids, name, im)
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state
