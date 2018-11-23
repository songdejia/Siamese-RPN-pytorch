# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-11-22 15:35:00
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-11-22 15:38:54
import os

path = '/home/song/srpn/dataset/vid/ILSVRC2015_VID_train_0000'
maxlen = 0
minlen = 40
for root, dirs, files in os.walk(path):
	if len(files) == 0:
		continue
	maxlen = max(len(files), maxlen)
	minlen = min(len(files), minlen)

	print(maxlen)
	print(minlen)