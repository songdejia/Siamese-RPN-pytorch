# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-30 16:51:20
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-30 16:52:32
import os
root = '/home/song/workspace/srpn/OTB2015/otb15'

os.chdir(root)
for file in os.listdir(root):
	os.system('unzip {}'.format(file))
