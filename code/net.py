# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-11-05 11:16:24
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-11-05 11:18:14
# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
import torch


class SiamRPNBIG(nn.Module):
    def __init__(self, feat_in=512, feature_out=512, anchor=5):
        super(SiamRPNBIG, self).__init__()
        self.anchor = anchor
        self.feature_out = feature_out
        self.featureExtract = nn.Sequential(
            nn.Conv2d(3, 192, 11, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 512, 5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(512, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 512, 3),
            nn.BatchNorm2d(512),
        )
        self.conv_r1 = nn.Conv2d(feat_in, feature_out*4*anchor, 3)
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3)
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out*2*anchor, 3)
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)
        self.regress_adjust = nn.Conv2d(4*anchor, 4*anchor, 1)

        self.r1_kernel = []
        self.cls1_kernel = []

    def forward(self, x):
        x_f = self.featureExtract(x)       #[1, 512, 22, 22]
        r_detection = self.conv_r2(x_f)    #[1, 512, 20, 20]
        c_detection = self.conv_cls2(x_f)  #[1, 512, 20, 20]


        r = F.conv2d(r_detection, self.r1_kernel)                   #1,  20, 17, 17
        c = F.conv2d(c_detection, self.cls1_kernel)                 #1,  10, 17, 17
        ra= self.regress_adjust(r)
        """
        print('Detection')
        print(r_detection.shape)                                    #1, 512, 20, 20
        print(c_detection.shape)                                    #1, 512, 20, 20
        print('KERNEL')
        print('r1_kernel shape {}'.format(self.r1_kernel.shape))    #20, 512, 4, 4 20个通道为512的4*4卷积核
        print('c1_kernel shape {}'.format(self.cls1_kernel.shape))  #10, 512, 4, 4
        print(ra.shape)
        """
        return ra, c

    def temple(self, z):
        z_f = self.featureExtract(z)
        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        kernel_size = r1_kernel_raw.data.size()[-1]
        self.r1_kernel = r1_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size)


if __name__ == '__main__':
    t = torch.autograd.Variable(torch.ones((1, 3, 127, 127)))
    d = torch.autograd.Variable(torch.ones((1, 3, 255, 255)))
    net = SiamRPNBIG()
    net.temple(t)
    net(d)
