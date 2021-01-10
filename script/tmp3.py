import os
import shutil
import random

src_path = '/home/liqiang/Documents/FOG_DETECTION/Data/frosi_data/FROSI'
valid_path = '/home/liqiang/Desktop/valid_frosi.txt'
train_path = '/home/liqiang/Desktop/train_frosi.txt'
test_path = '/home/liqiang/Desktop/test_frosi.txt'
train_file = open(train_path, 'w')
valid_file = open(valid_path, 'w')
test_file = open(test_path, 'w')

ttlist = os.listdir(src_path)
num  = len(ttlist)
random.shuffle(ttlist)
random.shuffle(ttlist)
k = 0
for i in ttlist:
    line = i + '\n'
    if k < num * 0.7:
        train_file.write(line)
    elif k < num * 0.8:
        valid_file.write(line)
    else:
        test_file.write(line)
    k += 1
train_file.close()
valid_file.close()
test_file.close()


