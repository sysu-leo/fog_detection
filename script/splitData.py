import os
path = '/home/liqiang/Documents/FOG_DETECTION/Data/FROSI/Fog'
path2 = '/home/liqiang/Documents/FOG_DETECTION/fog_detection/Dataset/train_frosi.txt'
path3 = '/home/liqiang/Documents/FOG_DETECTION/fog_detection/Dataset/test_frosi.txt'

file_1 = open(path2, 'w')
file_2 = open(path3, 'w')

tlist = [50, 100, 150, 200, 250, 300, 400]

for i in range(len(tlist)):
    path_new = os.path.join(path, str(tlist[i]))
    ttlist = os.listdir(path_new)
    al_number = 504
    for tt in range(al_number):
        if tt < al_number* 0.8:
            line_1 = ttlist[tt] + ' ' + str(i) + '\n'
            file_1.write(line_1)
        else:
            line_2 = ttlist[tt] + ' ' + str(i) + '\n'
            file_2.write(line_2)

file_1.close()
file_2.close()
