import os
import shutil

path = '/home/liqiang/Documents/lstron_data/data/SCENE_1/1_new'
dst_path = '/home/liqiang/Documents/lstron_data/data/SCENE_1/1_new_1'
count = 0
cls = '1'
for i in os.listdir(path):
    try:
        t = i.strip().split('.')[0].split('-')[-1]
        t = int(t)
        newname = cls + '-' + str(count) + '-' + str(t) + '.jpg'
        shutil.copy(os.path.join(path, i), os.path.join(dst_path, newname))
        count += 1
    except:
        print('error')