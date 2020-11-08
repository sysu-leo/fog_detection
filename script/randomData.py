import os
import random

path1 = '/home/liqiang/Documents/FOG_DETECTION/fog_detection/Dataset/train_frosi.txt'
path2 = '/home/liqiang/Documents/FOG_DETECTION/fog_detection/Dataset/test_frosi.txt'

file1 = open(path1, 'r')
file2 = open(path2, 'r')

tlist1 = file1.readlines()
tlist2 = file2.readlines()

random.shuffle(tlist1)
random.shuffle(tlist1)

random.shuffle(tlist2)
random.shuffle(tlist2)

file1.close()
file2.close()

file_1  = open(path1, 'w')
file_2 = open(path2, 'w')

for i in tlist1:
    file_1.write(i)

for j in tlist2:
    file_2.write(j)
file_1.close()
file_2.close()
