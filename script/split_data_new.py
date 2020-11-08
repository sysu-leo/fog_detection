import os
import shutil

path = '/home/liqiang/Documents/FOG_DETECTION/fog_detection/Dataset/train.txt'
file = open(path, 'r')
path2 = '/home/liqiang/Documents/FOG_DETECTION/fog_detection/Dataset/new_train.txt'
file2 = open(path2, 'w')
tlist = file.readlines()

def function(num, level):
    if(num > 10000):
        print('error')
    if level == 0 or level == 1:
        return 1.0
    elif level == 2:
        if num >= 200 and num < 500:
            return float(num / 500.0)
        elif num >= 500 or num < 200:
            return 350.0/500.0
    elif level == 3:
        print("{}, {}".format(num, level))
        if num < 200 and num >= 100:
            return num / 500.0
        elif num >200 or num < 100:
            return 150.0/500.0
    elif level == 4:
        if num <100:
            return float(num / 500.0)
        else:
            return 75.0/500.0
    elif level == 5:
        return 1.0

for i in tlist:
    ttlist = i.strip().split(' ')
    line = ''
    vis = 0.0
    if len(ttlist) == 2 :
        level = int(ttlist[-1])
        t_list = ttlist[0].split('.')
        num = int(t_list[0].split('-')[-1])
        vis = function(float(num), level)
        # print("{}, {}".format(num, level))
        line = i.strip() + ' ' + str(vis)
    if len(ttlist) == 3 :
        level = int(ttlist[-1])
        t_list = ttlist[1].split('.')
        tmp = t_list[0].split('-')[-1]
        num = int(tmp.strip().split('_')[-1])
        vis = function(float(num), level)
        # print("{}, {}".format(num, level))
        line = i.strip() + ' ' + str(vis)
    if len(ttlist) == 4:
        level = int(ttlist[-1])
        num = int(ttlist[0].split('-')[-1])
        vis = function(num, level)
        # print("{}, {}".format(num, level))
        line = i.strip() + ' ' + str(vis)
    file2.write(line+ '\n')
