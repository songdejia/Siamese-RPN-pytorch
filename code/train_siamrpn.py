# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-11-09 10:06:59
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-11-21 09:56:37
import os
import os.path as osp
import random
import time
import sys; sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
from PIL import Image, ImageOps, ImageStat, ImageDraw
from data_loader import TrainDataLoader
from net import SiameseRPN, SiameseRPN_bn
from torch.nn import init

parser = argparse.ArgumentParser(description='PyTorch SiameseRPN Training')

parser.add_argument('--train_path', default='/home/song/srpn/dataset/vot13', metavar='DIR',help='path to dataset')

parser.add_argument('--weight_dir', default='/home/song/srpn/weight', metavar='DIR',help='path to weight')

parser.add_argument('--checkpoint_path', default=None, help='resume')

parser.add_argument('--max_epoches', default=10000, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--max_batches', default=0, type=int, metavar='N', help='number of batch in one epoch')

parser.add_argument('--init_type',  default='xavier', type=str, metavar='INIT', help='init net')

parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')

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
    closses, rlosses, tlosses = AverageMeter(), AverageMeter(), AverageMeter()
    for epoch in range(start, args.max_epoches):
        cur_lr = adjust_learning_rate(args.lr, optimizer, epoch, gamma=0.1)
        index_list = range(data_loader.__len__()) 
        #for example in range(args.max_batches):
        for example in range(900):
            ret = data_loader.__get__(random.choice(index_list)) 
            template = ret['template_tensor'].cuda()
            detection= ret['detection_tensor'].cuda()
            pos_neg_diff = ret['pos_neg_diff_tensor'].cuda() if ret['pos_neg_diff_tensor'] is not None else None
            
            cout, rout = model(template, detection)
            
            predictions = (cout, rout)
            targets = pos_neg_diff

            area = ret['area_target_in_resized_detection']
            num_pos = len(np.where(pos_neg_diff == 1)[0])
            if area == 0 or num_pos == 0 or pos_neg_diff is None:
                continue



            closs, rloss, loss, reg_pred, reg_target, pos_index, neg_index = criterion(predictions, targets)
            
            # debug for class
            cout = cout.squeeze().permute(1, 2, 0).reshape(-1, 2)
            cout = cout.cpu().detach().numpy()
            print(cout.shape)
            score = 1/(1 + np.exp(cout[:,0]-cout[:,1]))
            print(score[pos_index])
            print(score[neg_index])
            #time.sleep(1)

            # debug for reg
            tmp_dir = '/home/song/srpn/tmp/visualization/7_train_debug_pos_anchors'
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            detection = ret['detection_cropped_resized'].copy()
            draw = ImageDraw.Draw(detection)
            pos_anchors = ret['pos_anchors'].copy()
            
            # pos anchor的回归情况
            x = pos_anchors[:,0] + pos_anchors[:, 2] * reg_pred[pos_index, 0].cpu().detach().numpy()
            y = pos_anchors[:,1] + pos_anchors[:, 3] * reg_pred[pos_index, 1].cpu().detach().numpy()
            w = pos_anchors[:,2] * np.exp(reg_pred[pos_index, 2].cpu().detach().numpy())
            h = pos_anchors[:,3] * np.exp(reg_pred[pos_index, 3].cpu().detach().numpy())
            x1s, y1s, x2s, y2s = x - w//2, y - h//2, x + w//2, y + h//2
            for i in range(2):
                x1, y1, x2, y2 = x1s[i], y1s[i], x2s[i], y2s[i]
                draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='red') #predict
            
            # 应当的gt
            x = pos_anchors[:,0] + pos_anchors[:, 2] * reg_target[pos_index, 0].cpu().detach().numpy()
            y = pos_anchors[:,1] + pos_anchors[:, 3] * reg_target[pos_index, 1].cpu().detach().numpy()
            w = pos_anchors[:,2] * np.exp(reg_target[pos_index, 2].cpu().detach().numpy())
            h = pos_anchors[:,3] * np.exp(reg_target[pos_index, 3].cpu().detach().numpy())
            x1s, y1s, x2s, y2s = x - w//2, y-h//2, x + w//2, y + h//2
            for i in range(2):
                x1, y1, x2, y2 = x1s[i], y1s[i], x2s[i], y2s[i]
                draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='green') #gt

            # 找分数zui da de,
            m_indexs = np.argsort(score)[::-1][:5]
            for m_index in m_indexs:
                diff = reg_pred[m_index].cpu().detach().numpy()
                anc  = ret['anchors'][m_index]
                x = anc[0] + anc[0] * diff[0]
                y = anc[1] + anc[1] * diff[1]
                w = anc[2]*np.exp(diff[2])
                h = anc[3]*np.exp(diff[3])
                x1, y1, x2, y2 = x - w//2, y - h//2, x + w//2, y + h//2
                draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=2, fill='black')


            save_path = osp.join(tmp_dir, 'epoch_{:04d}_{:04d}_{:02d}.jpg'.format(epoch, example, i))
            detection.save(save_path)

            closs_ = closs.cpu().item()
            if np.isnan(closs_): 
               sys.exit(0)

            #loss = closs + rloss
            closses.update(closs.cpu().item())
            rlosses.update(rloss.cpu().item())
            tlosses.update(loss.cpu().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #time.sleep(1)

            print("Epoch:{:04d} example:{:06d} lr:{:.7f} closs:{:.6f} \t rloss:{:.6f} \t tloss:{:.6f}".format(epoch, example+1, cur_lr, closses.avg, rlosses.avg, tlosses.avg ))


    
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

    def forward(self, predictions, targets):
        print('+++++++++++++++++++++++++++++++++++')
        cout, rout = predictions

        """ class """

        class_pred   = cout.squeeze().permute(1,2,0).reshape(-1, 2)
        class_target = targets[:, 0].long()
        pos_index = list(np.where(class_target == 1)[0])
        neg_index = list(np.where(class_target == 0)[0])
        class_target = class_target[pos_index + neg_index]
        class_pred   = class_pred[pos_index + neg_index]

        closs = F.cross_entropy(class_pred, class_target, size_average=False, reduce=False)
        closs = torch.div(torch.sum(closs[np.where(class_target != -100)]), 64)
        
        reg_pred = rout.view(-1, 4)
        reg_target = targets[:, 1:] #[1445, 4]
        rloss = F.smooth_l1_loss(reg_pred, reg_target, size_average=False, reduce=False)
        rloss = torch.div(torch.sum(rloss[np.where(class_target == 1)]), 16)


        #debug vis pos anchor
        loss = closs + rloss
        return closs, rloss, loss, reg_pred, reg_target, pos_index, neg_index



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
 

