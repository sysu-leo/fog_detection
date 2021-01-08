import os

path = '/home/liqiang/Desktop/3'
file = open('/home/liqiang/Desktop/3.txt', 'w')

for i in os.listdir(path):
    line = path + '/' + i +'\n'
    file.write(line)
file.close()
