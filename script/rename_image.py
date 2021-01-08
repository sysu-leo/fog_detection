import os
import shutil

path = '/home/liqiang/Desktop/3'
dst_path = '/home/liqiang/Desktop/3'
count = 0
cls = '3'
for i in os.listdir(path):
    try:
        t = i.strip().split('.')[0].split('-')[-1]
        newname = cls + '-' + str(count) + '-' + t + '.jpg'
        shutil.move(os.path.join(path, i), os.path.join(dst_path, newname))
        count += 1
    except:
        print('error')