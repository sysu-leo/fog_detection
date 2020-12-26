import shutil
import random

file1 = open('/home/liqiang/Documents/lstron_data/data/valid.txt', 'r')
ttlist1 = file1.readlines()

file1.close()

file2 = open('/home/liqiang/Documents/lstron_data/data/train.txt', 'r')
ttlist2 = file2.readlines()
file2.close()

file1 = open('/home/liqiang/Documents/lstron_data/data/valid.txt', 'w')
file2 = open('/home/liqiang/Documents/lstron_data/data/train.txt', 'w')

random.shuffle(ttlist1)
random.shuffle(ttlist1)
random.shuffle(ttlist1)

random.shuffle(ttlist2)
random.shuffle(ttlist2)
random.shuffle(ttlist2)

for i in ttlist1:
    file1.write(i)

for j in ttlist2:
    file2.write(j)

file1.close()
file2.close()
