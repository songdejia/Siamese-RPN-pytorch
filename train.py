# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-29 14:26:34
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-30 21:50:52

import torch
from torch import nn
from torch.autograd import Variable as V
import os
from axis import SmoothL1Loss
from axis import Myloss
#%%
def train_model(dataloader, model, optimizer, lmbda, scheduler, num_epochs, pth_dir, use_gpu):
    if not os.path.exists(pth_dir):
        os.makedirs(pth_dir)
    dirlist = os.listdir(pth_dir)
    if (dirlist):
#        del dirlist[dirlist.index('record.txt')]
        l = [int(i.split('.')[0].split('_')[-1]) for i in dirlist]
        former_epoch = max(l)
        model.load_state_dict(torch.load(pth_dir+'/epoch_'+str(former_epoch)+'.pth'))
        print('former_epoch %d loaded.' % former_epoch)
    else:
        former_epoch = 0
        print('first train begin.')
    for epoch in range(former_epoch+1, num_epochs+1):
            print('-' * 20)
            print('Epoch {}/{}'.format(epoch, num_epochs))

#        for phase in ['train', 'valid']:
            phase = 'train'
            epoch_loss = 0
            epoch_closs = 0
            epoch_rloss = 0
            
            if phase == 'train':
                print('-----train-----')
                if scheduler:
                    scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                print('-----valid-----')
                model.train(False)  # Set model to evaluate mode

            phase = 'train'
            for i, tvdata in enumerate(dataloader[phase]):
                template, detection, clabel, rlabel, pcc, ratio = tvdata
                """"""
#                template, detection, clabel, rlabel, pcc, ratio = template.squeeze(), detection.squeeze(), clabel.squeeze(), rlabel.squeeze(), pcc.squeeze(), ratio.squeeze()
#                template, detection, clabel, rlabel, pcc, ratio = template.numpy(), detection.numpy(), clabel.numpy(), rlabel.numpy(), pcc.numpy(), ratio.numpy()
#                import cv2
#                import numpy as np
#                import math
#                from axis import xywh_to_x1y1x2y2
#                template = np.transpose(template,(1,2,0))
#                template = cv2.cvtColor(template, cv2.COLOR_RGB2BGR)
#                cv2.imshow('img', template)
#                cv2.waitKey(0)
#
#                detection = np.transpose(detection,(1,2,0))
#                detection = cv2.cvtColor(detection, cv2.COLOR_RGB2BGR)
##                cv2.imshow('img', detection)
##                cv2.waitKey(0)
##
#                a = 64
#                s = a**2
#                r = [[3*math.sqrt(s/3.),math.sqrt(s/3.)], [2*math.sqrt(s/2.),math.sqrt(s/2.)], [a,a], [math.sqrt(s/2.),2*math.sqrt(s/2.)], [math.sqrt(s/3.),3*math.sqrt(s/3.)]]
#                r = [list(map(round, i)) for i in r]
#                
#                loc1 = np.where(clabel > 0.5)
##                img = cv2.imread('./lq/JPEGImages/'+os.listdir('./lq/JPEGImages/')[i])
#                for where in range(len(loc1[0])):
#                    loc = [loc1[0][where], loc1[1][where], loc1[2][where]]
#
#                    anchor = [7+15*loc[1], 7+15*loc[2]] + r[loc[0]] #根据loc确定anchor
#                    "根据loc确定对anchor的修正："
#                    reg = [rlabel[loc[0]*4, loc[1], loc[2]], rlabel[loc[0]*4+1, loc[1], loc[2]], rlabel[loc[0]*4+2, loc[1], loc[2]], rlabel[loc[0]*4+3, loc[1], loc[2]]]
#                    "根据anchor及reg确定proposals"
#                    pro = [anchor[0]+reg[0]*anchor[2], anchor[1]+reg[1]*anchor[3], anchor[2]*math.exp(reg[2]), anchor[3]*math.exp(reg[3])]
##                    pro = anchor
##                    "把在255X255中的proposals转换成原图的对应位置"
##                    pro2 = [pro[0]*ratio+pcc[2]-pcc[0], pro[1]*ratio+pcc[3]-pcc[1], pro[2]*ratio, pro[3]*ratio]
#                    list1 = xywh_to_x1y1x2y2(pro)
#                    list1 = list(map(lambda x:int(round(x)), list1))
#                    cv2.rectangle(detection, (list1[0],list1[1]), (list1[2],list1[3]), (0,255,0), 1)
#                cv2.imshow('img', detection)
#                cv2.waitKey(0)
#                detection = Image.fromarray(cv2.cvtColor(detection,cv2.COLOR_BGR2RGB))
#                detection.save('./tmp/'+str(i)+'.jpg')
#                cv2.imwrite('./tmp/'+str(i)+'.jpg', detection, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                """"""
                if use_gpu:
                    target = torch.zeros(clabel.shape).cuda()+1
                    template = V(template.cuda())
                    detection = V(detection.cuda())
                    clabel = V(clabel.cuda())
                    rlabel = V(rlabel.cuda())
                    model = model.cuda()
                else:
                    target = torch.zeros(clabel.shape)+1
                    template = V(template)
                    detection = V(detection)
                    clabel = V(clabel)
                    rlabel = V(rlabel)

                optimizer.zero_grad()

                # forward
                coutput, routput = model(template, detection)
#                coutput, routput, clabel, rlabel = coutput.squeeze(), routput.squeeze(), clabel.squeeze(), rlabel.squeeze()
                coutput, clabel = coutput.squeeze(), clabel.squeeze()
                coutput = coutput.view(5, 2, 17, 17)              # Batch*k*2*17*17
                
#                routput0 = routput[0].data.numpy()
#                rlabel0 = rlabel[0].data.numpy()
                closs = nn.CrossEntropyLoss()(coutput, clabel)
                
                rloss = SmoothL1Loss(use_gpu = use_gpu)(clabel, target, routput, rlabel)
#                rloss = nn.SmoothL1Loss()(routput, rlabel)
                loss = Myloss()(coutput, clabel, target, routput, rlabel, lmbda)                

#                loss = closs + lmbda * rloss
                loss2 = torch.add(closs, lmbda, rloss)
                epoch_loss += loss2.data.item()
                epoch_closs += closs.data.item()
                epoch_rloss += rloss.data.item()
#                epoch_rloss += 0
                # backward + optimize only if in training
                if phase == 'train':                    
                    loss.backward()
                    optimizer.step()
                
                # statistics
                
#                top1num, top1acc = accuracy(outputs, labels, 1)
#                top3num, top3acc = accuracy(outputs, labels, 3)
                
#                epoch_top1num += top1num
#                epoch_top3num += top3num
                
                if (phase == 'train'):
                    if(i+1 == 2 or (i+1) % 100 == 0):
                        print('batch %d, train loss:%.6f' % (i+1, loss.data.item()))
#                        duration = time.time() - since
#                        print('step %d in %.0f seconds. loss: %.6f' % (i+1, duration, loss.data[0]))
#                        print(' * top1acc:{top1acc:.6f}; top3acc:{top3acc:.6f}'
#                                  .format(top1acc=top1acc, top3acc=top3acc))
                    if (i+1 == len(dataloader[phase])):
                        print('train loss:%.6f' % (epoch_loss/len(dataloader[phase])))
                        print('closs:%.6f' % (epoch_closs/len(dataloader[phase])))
                        print('rloss:%.6f' % (epoch_rloss/len(dataloader[phase])))
#                        with open(RECORD_FILE, 'a') as f:
#                            f.write('-'*20 + '\nEpoch %d/%d\n' % (epoch,num_epochs))
#                            f.write('Epoch %d: loss:%.6f; top1acc:%.6f; top3acc:%.6f\n' 
#                                % (epoch, epoch_loss/len(dataloader), epoch_top1num/len(dataset_train), epoch_top3num/len(dataset_train)))
#                elif (phase == 'valid'):
#                    if (i+1 == len(valid_dataloader)):
#                        print('\nvalid loss:%.6f;\ntop1acc:%.6f; top3acc:%.6f' 
#                              % (epoch_loss/len(valid_dataloader), epoch_top1num/len(dataset_valid), epoch_top3num/len(dataset_valid)))
#                        with open(RECORD_FILE, 'a') as f:
#                            f.write('Epoch %d: loss:%.6f; top1acc:%.6f; top3acc:%.6f\n' 
#                                % (epoch, epoch_loss/len(valid_dataloader), epoch_top1num/len(dataset_valid), epoch_top3num/len(dataset_valid)))
                    
            # deep copy the model
#        if epoch_acc > best_acc:
#                best_acc = epoch_acc
#                best_model_wts = model_conv.state_dict()

            torch.save(model.state_dict(), (pth_dir + 'epoch_%d.pth')% epoch)
#            print('current model saved to epoch_%d.pth' % epoch)
    
#%%
from SRPN import SiameseRPN
#from data import dataloader
from data_otb import dataloader
import torch.optim as optim
from torch.optim import lr_scheduler
#%%
if __name__ == '__main__':
    
    model = SiameseRPN()
    
    params = []
#    params += list(model.features[0].parameters())
#    params += list(model.features[3].parameters())
#    params += list(model.features[6].parameters())
    params += list(model.features[8].parameters())
    params += list(model.features[10].parameters())
    params += list(model.conv1.parameters())
    params += list(model.conv2.parameters())
    params += list(model.conv3.parameters())
    params += list(model.conv4.parameters())
    
    optimizer = optim.Adam(params, lr=1e-3, eps=1e-8, weight_decay=0)
#    optimizer = optim.SGD(params, lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    
    train_model(
            dataloader = dataloader
            ,
            model = model
            , 
            optimizer = optimizer
            , 
#            scheduler = scheduler
            scheduler = None
            , 
            lmbda = 1
            ,
            num_epochs = 100
            , 
            pth_dir = './pth_OTB2015/'
            ,
            use_gpu = True
            )
#%%


