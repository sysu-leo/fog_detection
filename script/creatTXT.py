import os

path = '/home/liqiang/Documents/lstron_data/data_new/SCENE_1/4'
file = open('/home/liqiang/Documents/lstron_data/data_new/SCENE_1/4.txt', 'w')

for i in os.listdir(path):
    line = path + '/' + i +'\n'
    file.write(line)
file.close()
