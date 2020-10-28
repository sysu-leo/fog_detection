import os

path = '/home/liqiang/Desktop/DIP_homework2/libsvm-3.24/data/train.txt'
path2 = '/home/liqiang/Desktop/DIP_homework2/libsvm-3.24/data/New_train.txt'
file = open(path, 'r')
file2 = open(path2, 'w')
for i in file.readlines():
    tlist = i.split(' ')
    if int(tlist[0])  <= 9:
        file2.write(i)
