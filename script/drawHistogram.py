import matplotlib.pyplot as plt
import os

path = '/home/liqiang/Documents/lstron_data/train.txt'
file = open(path, 'r')
tlist = []
for i in range(500):
    tlist.append(0)
for i in file.readlines():
    try:
        vis = i.split('.')[0].split('-')[-1]
        vis = int(vis)
        if vis >= 500:
            continue
        tlist[vis] += 1
    except:
        print('error')
plt.bar(range(500), tlist, fc='b')
plt.show()