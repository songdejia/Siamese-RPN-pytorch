# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-11-22 10:48:51
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-11-22 15:06:36
import os
import sys
vid_path = '/home/song/srpn/dataset/vid/ILSVRC2015_VID_train_0000'

def remove_space(l):
	ns = []
	for i in l:
		n = i.replace('\t', '')
		ns.append(n)
	return ns

def trans(l):
	ns = []
	for i in l:
		x1, y2, w, h = map(float, i.split(',')[:4])
		x2, y1 = x1 + w, y2 - h
		line = '{},{},{},{}\n'.format(x1, y1, w, h)
		ns.append(line)
	return ns

count = 0
for path, dirs, files in os.walk(vid_path):
	for filename in files:
		if filename.find('JPEG') != -1:
			newname = filename.split('.')[0]+'.jpg'
		elif filename.find('txt') != -1:
			newname = 'groundtruth.txt'
		else:
			newname = filename
		old = os.path.join(path, filename)
		new = os.path.join(path, newname)
		os.rename(old, new)

		if filename.find('txt') != -1:
			with open(new, 'r') as f:
				lines = f.readlines()
				ns = remove_space(lines)
				n  = trans(ns)

			with open(new, 'w') as f:
				f.writelines(n)

		count += 1
		print(count)