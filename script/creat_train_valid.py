import os
import shutil

src_path = '/home/dell/Documents/data/SCENE_1_slice'
valid_path = '/home/dell/Desktop/valid.txt'
train_path = '/home/dell/Desktop/train.txt'
test_path = '/home/dell/Desktop/test.txt'
train_file = open(train_path, 'w')
valid_file = open(valid_path, 'w')
test_file = open(test_path, 'w')

for i in os.listdir(src_path):
    new_path = os.path.join(src_path, i)
    tlist = os.listdir(new_path)
    num = len(tlist)
    for j in range(num):
        line = tlist[j] + '\n'
        if j < num * 0.7:
            train_file.write(line)
        elif j < num * 0.8:
            valid_file.write(line)
        else:
            test_file.write(line)

train_file.close()
valid_file.close()
test_file.close()


