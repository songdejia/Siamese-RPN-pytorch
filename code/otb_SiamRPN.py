# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-11-05 16:04:00
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-11-05 20:52:00
import sys
import cv2  # imread
import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
import os
import time
from os.path import realpath, dirname, join
from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

# load net
net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG()
net.load_state_dict(torch.load(net_file))
net.eval().cuda()

OTB100_path = '/home/song/srpn/dataset/otb100'
result_path = '/home/song/srpn/result/'


# warm up
for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())
idx=0

names = [name for name in os.listdir('/home/song/srpn/dataset/otb100')]
# human4
# skating2
"""
for i, name in enumerate(names):
	print('idx == {:03d} name == {:10}'.format(i, name))
"""
for ids, x in enumerate(os.walk(OTB100_path)):
	it1, it2, it3 = x #it1 **/img   it2 []  it3 [img1, img2, ...] 
	if it1.rfind('img')!=-1 and len(os.listdir(it1)) > 50:#Python rfind() 返回字符串最后一次出现的位置(从右向左查询)，如果没有匹配项则返回-1。
		name = it1.split('/')[-2]
		imgpath=[]
		for inames in it3:
			imgpath.append(os.path.join(it1, inames))

		gtpath=os.path.join(OTB100_path, name, 'groundtruth_rect.txt')
		gt=(open(gtpath, 'r')).readline()
		if gt.find(',')!=-1:
			toks=map(float, gt.split(','))
		else:
			toks=map(float, gt.split('	'))

		cx=toks[0]+toks[2]*0.5
		cy=toks[1]+toks[3]*0.5
		w=toks[2]
		h=toks[3]
		target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
		
		# 第一张图作为模版
		im = cv2.imread(imgpath[0])
		state = SiamRPN_init(im, target_pos, target_sz, net)
		bbox=[];
		totalframe=len(imgpath)
		ttime=0

		# 第二张图直到结尾作为搜索图
		for imgs in imgpath[1:]:
			im = cv2.imread(imgs)
			ttime1=time.time()
			state = SiamRPN_track(state, im)  # track
    		res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
			bbox.append(res)
			ttime=ttime+time.time()-ttime1

		print ('Idx:%03d == total frame:%04d == speed:%03d fps'%(idx+1, totalframe, (totalframe-1)/ttime))
		saveroot=result_path+name+'.mat'
		scio.savemat(saveroot,{'bbox':bbox})
		idx=idx+1
