# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-30 16:56:21
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-30 16:57:54
import os
data_root = '/home/song/workspace/srpn/OTB2015/otb15'

for d in os.listdir(data_root):
	dir_path = os.path.join(data_root, d)
	os.chdir(dir_path)
	os.system('mkdir label')
	gt_dir_path = os.path.join(dir_path, 'label')
	if not os.path.exists(gt_dir_path):
		os.path.mkdir(gt_dir_path)


	gt_path = os.path.join(dir_path, 'groundtruth_rect.txt')
	with open(gt_path, 'r') as f:
		contents = f.readlines()

	for idx, line in enumerate(contents):
		print(line.strip('\n'))
		name = '{:04d}.txt'.format(idx+1)
		new_gt_path = os.path.join(gt_dir_path, name)
		with open(new_gt_path, 'w') as f:
			f.write(line)