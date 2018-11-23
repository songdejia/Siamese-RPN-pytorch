# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-11-22 17:06:59
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-11-22 17:07:51
from PIL import Image

path = '/home/song/srpn/dataset/vid/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00139005/000191.jpg'

im = Image.open(path)
im.show()