import os
import shutil

src_path = '/mnt/b2243f03-6052-4c66-8b36-9241f1f5d31b/SCENE_1_ok'
valid_path = '/home/liqiang/Desktop/valid.txt'
train_path = '/home/liqiang/Desktop/train.txt'
test_path = '/home/liqiang/Desktop/test.txt'
train_file = open(train_path, 'w')
valid_file = open(valid_path, 'w')
test_file = open(test_path, 'w')

for i in os.listdir(src_path):
    new_path = os.path.join(src_path, i)
    tlist = os.listdir(new_path)
    for j in range(len(tlist)):
        line = tlist[j] + '\n'
        if j < 1200 :
            train_file.write(line)
        elif j < 1300:
            valid_file.write(line)
        elif j < 1400:
            test_file.write(line)
        else:
            break
train_file.close()
valid_file.close()
test_file.close()


