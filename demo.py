# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-30 22:23:22
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-30 23:19:36
import os
import sys 
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from axis import Myloss
from data_otb import transformed_dataset_train as dataset 
from SRPN import SiameseRPN

model = SiameseRPN()
optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=1e-8, weight_decay=0)
criterion = Myloss()
#scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

length = dataset.__len__()
for i in range(length):
    template, detection, clabel, rlabel, pcc, rati = dataset.__getitem__(i)
    
    clabel = torch.from_numpy(clabel).view(1, -1)#[1, 5, 17, 17]
    rlabel = torch.from_numpy(rlabel).view(1, -1, 4)#[1, 20, 17, 17]
    _, rpn_match_index = np.where(clabel.numpy() == 1)

    target = torch.zeros(clabel.shape)+1
    coutput, routput = model(template, detection)
    coutput = coutput.view(1, 2, -1)
    routput = routput.view(1, -1, 4)
    routput = routput[0, rpn_match_index, :]
    rlabel  =  rlabel[0, rpn_match_index, :]
    #loss = Myloss()(coutput, clabel, target, routput, rlabel, lmbda)
    
    criterion_c = nn.CrossEntropyLoss()
    criterion_r = nn.SmoothL1Loss()
    loss_c = criterion_c(coutput, clabel)
    loss_r = criterion_r(routput, rlabel)
    #print(coutput.shape)
    #print(clabel.shape)
    #print(routput.shape)
    #print(rlabel.shape)
    print('index {:3d} loss_c {:.3f} loss_r {:.3f}'.format(i, loss_c.item()/64, loss_r.item()/16))  
    """
	bug here

    """
    sys.exit(0)
