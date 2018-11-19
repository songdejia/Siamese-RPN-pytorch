# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-11-09 10:06:59
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-11-19 17:18:45
import os
import random
import sys; sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
from data_loader import TrainDataLoader
from net import SiameseRPN
from torch.nn import init

parser = argparse.ArgumentParser(description='PyTorch SiameseRPN Training')

parser.add_argument('--train_path', default='/home/song/srpn/dataset/vot2013', metavar='DIR',help='path to dataset')

parser.add_argument('--weight_dir', default='/home/song/srpn/weight', metavar='DIR',help='path to weight')

parser.add_argument('--checkpoint_path', default=None, help='resume')

parser.add_argument('--max_epoches', default=100, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--max_batches', default=0, type=int, metavar='N', help='number of batch in one epoch')

parser.add_argument('--init_type',  default='xavier', type=str, metavar='INIT', help='init net')

parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='momentum', help='momentum')

parser.add_argument('--weight_decay', '--wd', default=5e-5, type=float, metavar='W', help='weight decay (default: 1e-4)')

def main():
    """ train dataloader """
    args = parser.parse_args()
    data_loader = TrainDataLoader(args.train_path, check = False)
    if not os.path.exists(args.weight_dir):
        os.makedirs(args.weight_dir)

    """ compute max_batches """
    for root, dirs, files in os.walk(args.train_path):
        for dirname in dirs:
            dir_path = os.path.join(root, dirname)
            args.max_batches += len(os.listdir(dir_path))
    inter = args.max_batches//10
    print('Max batches:{} in one epoch '.format(args.max_batches))

    """ Model on gpu """
    model = SiameseRPN()
    model = model.cuda()
    cudnn.benchmark = True

    """ loss and optimizer """
    criterion = MultiBoxLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)

    """ load weights """
    init_weights(model)
    if not args.checkpoint_path == None:
        assert os.path.isfile(args.checkpoint_path), '{} is not valid checkpoint_path'.format(args.checkpoint_path)
        try:
            checkpoint = torch.load(args.checkpoint_path)
            start = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            start = 0
            init_weights(model)
    else:
        start = 0

    """ train phase """
    closses, rlosses = AverageMeter(), AverageMeter()
    for epoch in range(start, args.max_epoches):
        cur_lr = adjust_learning_rate(args.lr, optimizer, epoch, gamma=0.1)
        index_list = range(data_loader.__len__()) 
        for example in range(args.max_batches):
            ret = data_loader.__get__(random.choice(index_list)) 
            template = ret['template_tensor'].cuda()
            detection= ret['detection_tensor'].cuda()
            pos_neg_diff = ret['pos_neg_diff_tensor'].cuda()
            cout, rout = model(template, detection)
            
            predictions = (cout, rout)
            targets = pos_neg_diff

            closs, rloss = criterion(predictions, targets)
            loss = closs + rloss
            closses.update(closs.cpu().item())
            rlosses.update(rloss.cpu().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if example % inter == 0:
                print("epoch:{:04d} example:{:06d} lr:{:.2f} closs:{:.2f}\trloss:{:.2f}".format(epoch, example, cur_lr, closses.avg, rlosses.avg))
    
        if epoch % 5 == 0 :
            file_path = os.path.join(args.weight_dir, 'epoch_{:04d}_weights.pth.tar'.format(epoch))
            state = {
            'epoch' :epoch+1,
            'state_dict' :model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }
            torch.save(state, file_path)

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        # this will apply to each layer
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('conv')!=-1 or classname.find('Linear')!=-1):
            if init_type=='normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')#good for relu
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    #print('initialize network with %s' % init_type)
    net.apply(init_func)



class MultiBoxLoss(nn.Module):
    def __init__(self):
        super(MultiBoxLoss, self).__init__()
        self.closs = torch.nn.CrossEntropyLoss()
        self.rloss = torch.nn.SmoothL1Loss()

    def forward(self, predictions, targets):
        cout, rout = predictions
        cout = cout.reshape(1, 2, -1)
        rout = rout.reshape(-1, 4)
        class_gt, diff = targets[:,0].unsqueeze(0).long(), targets[:,1:]
        closs = self.closs(cout, class_gt)#1,2,*  1,*

        pos_index = np.where(class_gt == 1)[1]
        if pos_index.shape[0] == 0:
            rloss = torch.FloatTensor([0]).cuda()
        else:
            rout_pos = rout[pos_index]
            diff_pos = diff[pos_index]
            
            #print(rout_pos)
            #print(diff_pos)
            rloss = self.rloss(rout_pos, diff_pos) #16
        return closs/64, rloss/16 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(lr, optimizer, epoch, gamma=0.1):
    """Sets the learning rate to the initial LR decayed 0.9 every 50 epochs"""
    lr = lr * (0.9 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    main()
 

