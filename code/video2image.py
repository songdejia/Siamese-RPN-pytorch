# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-11-09 10:13:23
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-11-09 17:21:16
import pandas as pd
import cv2
import os
import sys

workspace = os.path.abspath('./')
#### annotation #######################################################################
os.chdir(workspace)
yt_bb_ann_path = os.path.abspath('../youtube_BB/annotation')
if not os.path.exists(yt_bb_ann_path):
	os.makedirs(yt_bb_ann_path)
	os.chdir(yt_bb_ann_path)
	os.system('wget https://research.google.com/youtube-bb/yt_bb_detection_train.csv.gz') # here may need vpn in China
	os.system('wget https://research.google.com/youtube-bb/yt_bb_detection_validation.csv.gz') # here may need vpn in China


#### use script to download video #######################################################
os.chdir(workspace)
yt_bb_video_path = os.path.abspath('../youtube_BB/video')
yt_bb_script_path = os.path.abspath('../youtube_BB/youtube-bb-script/youtube-bb')
if not os.path.exists(yt_bb_video_path):
	os.makedirs(yt_bb_video_path)
	os.chdir(yt_bb_script_path)
	os.system('pip install -r requirements.txt')
	os.system('python3 download.py {} 6'.format(yt_bb_video_path))



### trans video2pic ######################################################################
f = pd.read_csv(os.path.join(yt_bb_ann_path, 'yt_bb_detection_train.csv'), header=None)
f.columns = ['youtube_id','timestamp_ms','class_id','class_name','object_id','object_presence','xmin','xmax','ymin','ymax']
#print(f['youtube_id'])

os.chdir(workspace)
for subdir_name in os.listdir(yt_bb_video_path):
	subdir_path = os.path.join(yt_bb_video_path, subdir_name)
	for mp4_name in os.listdir(subdir_path):
		mp4_path = os.path.join(subdir_path, mp4_name)
		video = cv2.VideoCapture(mp4_path) #VideoCapture()中参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
		#ret, frame = video.read() # cap.read()按帧读取视频，ret,frame是获cap.read()方法的两个返回值。其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False
		
		# 通过mp4_name寻找对应的bg info
		mp4_name = mp4_name.strip('.mp4')
		# test
		mp4_name = 'ej4xM04ipxM'
		print('mp4 name == {}'.format(mp4_name))
		

		############# 按帧读图片 ###############
		save_img_dir = os.path.abspath('../youtube_BB/image')
		save_img_subdir = os.path.join(save_img_dir, mp4_name)
		if not os.path.exists(save_img_subdir):
			os.makedirs(save_img_subdir)

		ret = True # by default the first frame is not the last one
		index = 0
		while ret:#当前一帧不是结尾,就读下一帧
			ret, frame = video.read()
			save_img_file = os.path.join(save_img_subdir, '{:04d}.jpg'.format(index))
			cv2.imwrite(save_img_file, frame)
			index += 1



		############# 按帧读gt  ###############
		lines = []		
		f_frames_of_mp4 = f.loc[f['youtube_id'] == mp4_name]
		save_gt_subdir = os.path.join(yt_bb_ann_path, mp4_name)
		if not os.path.exists(save_gt_subdir):
			os.makedirs(save_gt_subdir)
		save_gt_file = os.path.join(save_gt_subdir, 'groundtruth_rect.txt')
		for frame_index, frame_id in enumerate(f_frames_of_mp4['youtube_id']):
			info_of_this_frame = f_frames_of_mp4.iloc[frame_index]

			ids             = info_of_this_frame['youtube_id']
			time            = info_of_this_frame['timestamp_ms']
			class_id        = info_of_this_frame['class_id']
			class_name      = info_of_this_frame['class_name']
			object_id       = info_of_this_frame['object_id']
			object_presence = info_of_this_frame['object_presence']
			xmin            = info_of_this_frame['xmin']
			xmax            = info_of_this_frame['xmax']
			ymin            = info_of_this_frame['ymin']
			ymax            = info_of_this_frame['ymax']

			c_x             = (xmin + xmax)//2
			c_y             = (ymin + ymax)//2
			w               = xmax - xmin
			h               = ymax - ymin
			newline = "{},{},{},{}\n".format(c_x, c_y, w, h)
			lines.append(newline)

			print('Video {:12} == Frame index:{:03d} == time:{:06d} == class:{:8} == object_presence:{:8}'.format(frame_id, frame_index, time, class_name, object_presence)) 		
		with open(save_gt_file, 'w') as f:
			f.writelines(lines)













