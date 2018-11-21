# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-11-09 10:06:59
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-11-20 21:26:08
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
from PIL import Image, ImageOps, ImageStat, ImageDraw
from net import SiameseRPN
from torch.nn import init

parser = argparse.ArgumentParser(description='PyTorch SiameseRPN Training')

parser.add_argument('--train_path', default='/home/song/srpn/dataset/simple_vot13', metavar='DIR',help='path to dataset')

parser.add_argument('--weight_dir', default='/home/song/srpn/weight', metavar='DIR',help='path to weight')

parser.add_argument('--checkpoint_path', default='/home/song/srpn/weight/epoch_0060_weights.pth.tar', help='resume')

parser.add_argument('--max_epoches', default=100, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--max_batches', default=10000, type=int, metavar='N', help='number of batch in one epoch')

parser.add_argument('--init_type',  default='xavier', type=str, metavar='INIT', help='init net')

parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='momentum', help='momentum')

parser.add_argument('--weight_decay', '--wd', default=5e-5, type=float, metavar='W', help='weight decay (default: 1e-4)')

def main():
    """ dataloader """
    args = parser.parse_args()
    data_loader = TrainDataLoader(args.train_path, check = False)

    """ Model on gpu """
    model = SiameseRPN()
    model = model.cuda()
    cudnn.benchmark = True

    """ loss and optimizer """
    criterion = MultiBoxLoss()

    """ load weights """
    init_weights(model)
    if args.checkpoint_path == None:
        sys.exit('please input trained model')
    else:
        assert os.path.isfile(args.checkpoint_path), '{} is not valid checkpoint_path'.format(args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path)
        start = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    """ test phase """        
    index_list = range(data_loader.__len__()) 
    for example in range(args.max_batches):
        ret = data_loader.__get__(random.choice(index_list)) 
        template = ret['template_tensor'].cuda()
        detection= ret['detection_tensor'].cuda()
        pos_neg_diff = ret['pos_neg_diff_tensor'].cuda()
        cout, rout = model(template, detection) #[1, 10, 17, 17], [1, 20, 17, 17]

        cout = cout.reshape(-1, 2)
        rout = rout.reshape(-1, 4)
        cout = cout.cpu().detach().numpy()
        score = 1/(1 + np.exp(cout[:,1]-cout[:,0]))
        diff   = rout.cpu().detach().numpy() #1445
        
        num_proposals = 1
        score_64_index = np.argsort(score)[::-1][:num_proposals]

        score64 = score[score_64_index]
        diffs64 = diff[score_64_index, :] 
        anchors64 = ret['anchors'][score_64_index]
        proposals_x = (anchors64[:, 0] + anchors64[:, 2] * diffs64[:, 0]).reshape(-1, 1)
        proposals_y = (anchors64[:, 1] + anchors64[:, 3] * diffs64[:, 1]).reshape(-1, 1)
        proposals_w = (anchors64[:, 2] * np.exp(diffs64[:, 2])).reshape(-1, 1)
        proposals_h = (anchors64[:, 3] * np.exp(diffs64[:, 3])).reshape(-1, 1)
        proposals = np.hstack((proposals_x, proposals_y, proposals_w, proposals_h))

        d = os.path.join(ret['tmp_dir'], '6_pred_proposals')
        if not os.path.exists(d):
            os.makedirs(d)

        detection = ret['detection_cropped_resized']
        save_path = os.path.join(ret['tmp_dir'], '6_pred_proposals', '{:04d}_1_detection.jpg'.format(example))
        detection.save(save_path)

        template = ret['template_cropped_resized']
        save_path = os.path.join(ret['tmp_dir'], '6_pred_proposals', '{:04d}_0_template.jpg'.format(example))
        template.save(save_path)

        """ 可视化 """
        draw = ImageDraw.Draw(detection)
        for i in range(num_proposals):
            x, y, w, h = proposals_x[i], proposals_y[i], proposals_w[i], proposals_h[i]
            x1, y1, x2, y2 = x-w//2, y-h//2, x+w//2, y+h//2
            draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='red')
        """ save detection template proposals"""


        save_path = os.path.join(ret['tmp_dir'], '6_pred_proposals', '{:04d}_2_proposals.jpg'.format(example))
        detection.save(save_path)

        print('save at {}'.format(save_path))
            # restore




        """


            predictions = (cout, rout)
            targets = pos_neg_diff

            closs, rloss = criterion(predictions, targets)
            loss = closs + rloss
            closses.update(closs.cpu().item())
            rlosses.update(rloss.cpu().item())



            print("epoch:{:04d} example:{:06d} lr:{:.2f} closs:{:.2f}\trloss:{:.2f}".format(epoch, example, cur_lr, closses.avg, rlosses.avg))
    
        if epoch % 5 == 0 :
            file_path = os.path.join(args.weight_dir, 'epoch_{:04d}_weights.pth.tar'.format(epoch))
            state = {
            'epoch' :epoch+1,
            'state_dict' :model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }
            torch.save(state, file_path)
        """


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
 

