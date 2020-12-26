import os
import shutil

src_path = '/home/liqiang/Documents/lstron_data/data/SCENE_1_dark'
valid_path = '/home/liqiang/Documents/lstron_data/data/valid.txt'
train_path = '/home/liqiang/Documents/lstron_data/data/train.txt'

train_file = open(train_path, 'w')
valid_file = open(valid_path, 'w')

for i in os.listdir(src_path):
    new_path = os.path.join(src_path, i)
    tlist = os.listdir(new_path)
    for j in range(len(tlist)):
        line = tlist[j] + '\n'
        if j < 1200 :
            train_file.write(line)
        elif j < 1400:
            valid_file.write(line)
        else:
            break
train_file.close()
valid_file.close()


