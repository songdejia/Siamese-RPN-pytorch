# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-30 21:52:44
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-30 22:21:08

import sys
import torch
import math
from PIL import ImageStat, Image
from torchvision.transforms import functional as F2
from torch.nn import Module
from torch.nn import functional as F
def x1y1x2y2x3y3x4y4_to_x1y1wh(gtbox):
    x1, y1, x2, y2, x3, y3, x4, y4 = gtbox
    return [x1, y1, int(x2-x1), int(y3-y2)]


def x1y1x2y2x3y3x4y4_to_xywh(gtbox):
    x1, y1, x2, y2, x3, y3, x4, y4 = gtbox
    return [(x1+x2)//2, (y1+y3)//2, int(x2-x1), int(y3-y2)]


def x1y1x2y2_to_xywh(gtbox):
    return list(map(round, [(gtbox[0]+gtbox[2])/2., (gtbox[1]+gtbox[3])/2., gtbox[2]-gtbox[0], gtbox[3]-gtbox[1]]))


def xywh_to_x1y1x2y2(gtbox):
    return list(map(round, [gtbox[0]-gtbox[2]/2., gtbox[1]-gtbox[3]/2., gtbox[0]+gtbox[2]/2., gtbox[1]+gtbox[3]/2.]))


def x1y1wh_to_xywh(gtbox):
    """
    左上 x1, y1, w, h 转换到中心点 x, y, w, h
    """
    x1, y1, w, h = gtbox
    return [round(x1 + w/2.), round(y1 + h/2.), w, h]


def x1y1wh_to_x1y1x2y2(gtbox):
    x1, y1, w, h = gtbox
    return [x1, y1, x1+w, y1+h]

"""
class SmoothL1Loss(Module):
    def __init__(self, use_gpu):
        super (SmoothL1Loss, self).__init__()
        self.use_gpu = use_gpu
        return
    
    def forward(self, clabel, target, routput, rlabel):
        rloss = F.smooth_l1_loss(routput, rlabel, size_average=False, reduce=False)
        
            
        e = torch.eq(clabel.float(), target) 
        e = e.squeeze()
        e0,e1,e2,e3,e4 = e[0].unsqueeze(0),e[1].unsqueeze(0),e[2].unsqueeze(0),e[3].unsqueeze(0),e[4].unsqueeze(0)
        eq = torch.cat([e0,e0,e0,e0,e1,e1,e1,e1,e2,e2,e2,e2,e3,e3,e3,e3,e4,e4,e4,e4], dim=0).float()
        
        rloss = rloss.squeeze()
        rloss = torch.mul(eq, rloss)
        rloss = torch.sum(rloss)
        rloss = torch.div(rloss, eq.nonzero().shape[0]+1e-4)
        return rloss
"""

class Myloss(Module):
    def __init__(self):
        super (Myloss, self).__init__()
        return 
    
    def forward(self, coutput, clabel, target, routput, rlabel, lmbda):
        """
        clabel  -- [1, 20, 17, 17]
        routput -- [1, 10, 17, 17]
        rlabel  -- [1, 20, 17, 17]
        clabel  -- [1, 10, 17, 17]
        """
        closs = F.cross_entropy(coutput, clabel)

        rloss = F.smooth_l1_loss(routput, rlabel, size_average=False, reduce=False)
        
            
        e = torch.eq(clabel.float(), target) 
        e = e.squeeze()
        e0,e1,e2,e3,e4 = e[0].unsqueeze(0),e[1].unsqueeze(0),e[2].unsqueeze(0),e[3].unsqueeze(0),e[4].unsqueeze(0)
        eq = torch.cat([e0,e0,e0,e0,e1,e1,e1,e1,e2,e2,e2,e2,e3,e3,e3,e3,e4,e4,e4,e4], dim=0).float()
        
        rloss = rloss.squeeze()
        rloss = torch.mul(eq, rloss)
        rloss = torch.sum(rloss)
        rloss = torch.div(rloss, eq.nonzero().shape[0]+1e-4)
        
        loss = torch.add(closs, lmbda, rloss)
        return loss

def resize(img, size, interpolation=Image.BILINEAR):
    """
    ratio original.size[0]/size
    """
    assert img.size[0] == img.size[1]
    return img.resize((size, size), interpolation), img.size[0] / size

def point_center_crop(img, gtbox, area):
    """
    输入

    img instance 完整图片
    要求输出 以gt中央为中心的一个crop图片

    gtbox 中心形式 x0, y0, w, h
    
    Image instance 
    https://pillow.readthedocs.io/en/3.1.x/reference/Image.html
    """

    

    x, y, w, h = gtbox
    


    p = (w + h) / 2.
    a = math.sqrt((w + p) * (h + p))
    a *= area           #aa box的中心是xy, 边长a
    i = round(x - a/2.) #aa box左上角的x
    j = round(y - a/2.) #aa box左上角的y

    mean = tuple(map(round, ImageStat.Stat(img).mean))
    #ImageStat 属性 stat = ImageStat.Stat(img_instance) instance = image.open
    #1.stat.extrema 获取每个通道最大值最小值[(max, min), (max, min), (max, min)]
    #2.stat.count   获取每个通道像素个数
    #3.stat.sum     获取每个通道的像素值之和
    #4.stat.sum2    获取每个通道的像素值平方之和
    #5.stat.mean    获取每个通道的像素值的平均值
    #6.stat.median  获取每个通道的像素的中值
    #7.stat.rms     获取每个通道的像素值的均方根
    #8.stat.var     获取每个通道的方差值
    #9.stat.stddev  获取每个通道像素值的标准差值


    # 防止越界
    # left, top, right, bottom是越界长度
    if i < 0:
        left = -i
        i = 0
    else: 
        left = 0


    if j < 0:
        top = -j
        j = 0
    else: 
        top = 0

    if x+a/2. > img.size[0]:
        right = round(x+a/2.-img.size[0])
    else:
        right = 0

    if y+a/2. > img.size[1]:
        bottom = round(y+a/2.-img.size[1])
    else:
        bottom = 0
    
    #首先把越界区域补充上,left,top表示溢出长度(溢出的bbox的左上角点坐标),i,j表示未溢出时的左上角坐标(在原图内部的overlap的左上角)
    #最后模版是取得在原图内部的部分
    img = F2.pad(img, padding=(left, top, right, bottom), fill=mean, padding_mode='constant')   
    img = img.crop((i, j, i+round(a), j+round(a)))
    
    return img, [left, top, i, j]

def cosine_window(coutput1):
    math.cos()
    
    
    return

