# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-30 14:36:57
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-30 14:37:54
import os
import os.path as osp
data_root = '/home/song/workspace/srpn/OTB2015/otb15'

for d in os.listdir(data_root):
	if os.path.isdir(osp.join(data_root, d)):
		
		new_dir = osp.join(data_root, d)
		os.chdir(new_dir)
		if not os.path.exists(osp.join(data_root, 'img')):
			os.system('mkdir img && mv *.jpg ./img')

