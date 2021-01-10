import os
import shutil

path1 = '/home/liqiang/Documents/lstron_data/data/SCENE_1_slice/1'
path2 = '/home/liqiang/Documents/lstron_data/data/SCENE_1/1'
path3 = '/home/liqiang/Documents/lstron_data/data/SCENE_1/1_new'

for i in os.listdir(path1):
    print(i)
    img_path = os.path.join(path2, i)
    new_path = os.path.join(path3, i)
    shutil.copy(img_path, new_path)