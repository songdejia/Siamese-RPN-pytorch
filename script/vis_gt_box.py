# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-30 19:14:37
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-11-22 15:12:36
import os
import cv2
import sys
import numpy as np
from PIL import Image, ImageDraw
data_dir = '/home/song/srpn/dataset/vid/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00010013' # contain img and groundtruth.txt
save_dir = '/home/song/srpn/tmp/tmp_script'

imgnames = [name for name in os.listdir(data_dir) if name.find('.jpg') != -1]
imgnames = sorted(imgnames)

gt_path = os.path.join(data_dir, 'groundtruth.txt')
with open(gt_path, 'r') as f:
	lines = f.readlines()

for idx, i in enumerate(imgnames):
	print(idx)
	# gt
	line = lines[idx]
	x1, y1, w, h = [int(float(i)) for i in line.split(',')[:4]]
	x2 = x1 + w
	y2 = y1 + h 
	# img
	imgpath = os.path.join(data_dir, i)
	im = Image.open(imgpath)
	draw = ImageDraw.Draw(im)
	draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='red')

	save_path = os.path.join(save_dir, '{}.jpg'.format(idx))
	im.save(save_path)
