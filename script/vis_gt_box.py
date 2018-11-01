# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-30 19:14:37
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-30 20:01:20
import os
import cv2
import sys
import numpy as np
data_dir = '../OTB2015/otb15/subset/Human6'
save_dir = '../tmp/vis_background'

if not os.path.exists(save_dir):
	os.makedirs(save_dir)
# img shape fixed

for gtname in os.listdir(os.path.join(data_dir, 'label')):

	gtfile = os.path.join(data_dir, 'label', gtname)
	imgfile= os.path.join(data_dir, 'img'  , gtname.replace('txt', 'jpg'))
	newpath= os.path.join(save_dir, gtname.replace('txt', 'jpg'))
	img = cv2.imread(imgfile)[:,:,::-1]
	print(img.shape)
	with open(gtfile, 'r') as f:
		contents = f.read().strip('\n')
		if contents.find(',') == -1:
			x1, y1,  w, h = map(int, contents.split('\t'))
		else:
			x1, y1,  w, h = map(int, contents.split(','))

		#y1, x1, h, w = map(int, f.read().strip('\n').split('\t'))
		x2 = x1 + w
		y2 = y1
		x3 = x2
		y3 = y2 + h
		x4 = x1
		y4 = y3

		box = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
		cv2.polylines(img[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
		cv2.imwrite(newpath, img[:,:,::-1])
