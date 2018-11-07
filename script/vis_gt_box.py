# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-30 19:14:37
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-11-07 10:08:09
import os
import cv2
import sys
import numpy as np
data_dir = '/home/song/srpn/dataset/otb100'
save_dir = '/home/song/srpn/tmp/vis_gt'

"""
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
"""
for ids, name in enumerate(os.listdir(data_dir)):
	class_dir_path = os.path.join(data_dir, name)
	class_save_dir = os.path.join(save_dir, name)
	if not os.path.exists(class_save_dir):
		os.makedirs(class_save_dir)

	img_dir_path = os.path.join(class_dir_path, 'img')
	gt_file_path = os.path.join(class_dir_path, 'groundtruth_rect.txt')
	with open(gt_file_path, 'r') as f:
		cs = f.readlines()
	for ii, img_name in enumerate(sorted(os.listdir(img_dir_path))):
		if not len(os.listdir(img_dir_path)) == len(cs):
			print(name)
			continue
		newpath = os.path.join(class_save_dir, img_name)
		img_path = os.path.join(img_dir_path, img_name)
		
		img = cv2.imread(img_path)[:,:,::-1]
		contents = cs[ii].strip('\n').strip('\r')
		if contents.find(',') == -1:
			x1, y1, w, h = map(int, contents.split('\t'))
		else:
			x1, y1, w, h = map(int, contents.split(','))

		x2 = x1 + w
		y2 = y1
		x3 = x2
		y3 = y2 + h
		x4 = x1
		y4 = y3

		box = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
		cv2.polylines(img[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
		cv2.imwrite(newpath, img[:,:,::-1])