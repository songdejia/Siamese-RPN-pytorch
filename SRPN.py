# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-29 14:26:34
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-31 09:52:57
import sys
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}

class SiameseRPN(nn.Module):
    def __init__(self):
        super(SiameseRPN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3),
        )
        
        self.k = 5
        self.conv1 = nn.Conv2d(256, 2*self.k*256, kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 4*self.k*256, kernel_size=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu4 = nn.ReLU(inplace=True)

        self.cconv = nn.Conv2d(256, 2* self.k, kernel_size = 4, bias = False)
        self.rconv = nn.Conv2d(256, 4* self.k, kernel_size = 4, bias = False)
        
        self.reset_params()
        
    def reset_params(self):
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
            
    def forward(self, template, detection):
        """
        把template的类别,坐标的特征作为检测cconv和rconv的检测器
        把ckernel, rkernel转换到cconv, rconv
        """
        template = self.features(template)
        detection = self.features(detection)
        
        ckernal = self.conv1(template)
        ckernal = ckernal.view(2* self.k, 256, 4, 4)
        self.cconv.weight = nn.Parameter(ckernal)
        cinput = self.conv3(detection)
        coutput = self.cconv(cinput)
        
        rkernal = self.conv2(template)
        rkernal = rkernal.view(4* self.k, 256, 4, 4)
        self.rconv.weight = nn.Parameter(rkernal)
        rinput = self.conv4(detection)
        routput = self.rconv(rinput)
        
        return coutput, routput

    def resume(self, weight):
        checkpoint = torch.load(weight)
        self.load_state_dict(checkpoint)
        print('Resume checkpoint')

if __name__ == '__main__':
    model = SiameseRPN()

    template = torch.ones((1, 3, 127, 127))
    detection= torch.ones((1, 3, 255, 255))

    y1, y2 = model(template, detection)
    print(y1.shape) #[1, 10, 17, 17]
    print(y2.shape) #[1, 20, 17, 17]15

