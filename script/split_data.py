import os
import cv2
import numpy as np
import shutil

path1 = '/home/dell/Documents/fog_detection_day/Dataset/train.txt'
path2 = '/home/dell/Documents/fog_detection_day/Dataset/valid.txt'
# path3 = ''

file1 = open(path1, 'r')
file2 = open(path2, 'r')
# file3 = open(path3, 'r')

path_src = '/home/dell/Documents/fog_detection_day/Data/RGB'
path_dst = '/home/dell/Documents/fog_detection_day/split'
tlist1 = file1.readlines()
tlist2 = file2.readlines()
tlist3 = file2.readlines()

for line in tlist1:
    ttlist1 = line.strip().split(' ')
    class_id = int(ttlist1[-1])
    img_name = ''
    for i in range(len(ttlist1) - 2):
        img_name = img_name + ttlist1[i] + ' '
    img_name = img_name + ttlist1[-2]
    path_1 = os.path.join(path_src, img_name)
    path_2 = os.path.join(path_dst, ttlist1[-1], img_name)
    shutil.copy(os.path.join(path_src, img_name), os.path.join(path_dst, ttlist1[-1], img_name))

for line in tlist2:
    ttlist = line.strip().split(' ')
    class_id = int(ttlist[-1])
    img_name = ''
    for i in range(len(ttlist) - 2):
        img_name = img_name + ttlist[i] + ' '
    img_name = img_name + ttlist[-2]
    shutil.copy(os.path.join(path_src, img_name), os.path.join(path_dst, ttlist[-1], img_name))

# for line in tlist3:
#     ttlist = line.strip().split(' ')
#     class_id = int(ttlist[-1])
#     img_name = ''
#     for i in range(len(ttlist) - 2):
#         img_name = img_name + ttlist[i] + ' '
#     img_name = img_name + ttlist[-2]
#     shutil.copy(os.path.join(path_src, img_name), os.path.join(path_dst, ttlist[-1], img_name))




