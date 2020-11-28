import os
import shutil

path = '/home/dell/Documents/fog_detection_day/split'

tt = [10001, 500, 200, 100, 50, 30, 0]
for i in range(5):
    path_t = os.path.join(path, str(i))
    if not os.path.exists(os.path.join(path, str(i) +'_true')):
        os.mkdir(os.path.join(path, str(i) + '_true'))
    if not os.path.exists(os.path.join(path,str(i) +  '_false')):
        os.mkdir(os.path.join(path, str(i) + '_false'))
    tlist = os.listdir(path_t)
    for ii in tlist:
        print(ii+ '\n')
        num1 = (ii.strip().split(' ')[-1])
        num2 = ''
        if num1 != '副本.jpg'  and len(ii.strip().split(' ')) < 3:
            num2 =  num1.split('.')[-2]
        else:
            num2 = (ii.strip().split(' ')[0])
        num = num2.split('-')[-1]
        if int(num) > tt[i+1] and int(num) <= tt[i]:
            shutil.copy(os.path.join(path_t, ii), os.path.join(path, str(i) + '_true', ii))
        else:
            shutil.copy(os.path.join(path_t, ii), os.path.join(path, str(i) + '_false', ii))
