import os
import shutil


def gerRes(path):
    path2 = '../Data/test_data'
    path3= '../Result/'
    tt = path.split('/')[-1]
    ttt = tt.split('.')[0]
    path3 += ttt
    if not os.path.exists('../Result/'+ttt):
        os.system('mkdir ../Result/'+ttt)
        for i in range(6):
            os.system('mkdir ../Result/'+ttt +'/' + str(i))
            for j in range(6):
                os.system('mkdir ../Result/'+ttt+'/'+str(i)+'/'+str(j))
    file = open(path, 'r')



    llist = [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0,0, 0],
             [0, 0, 0, 0,0, 0],
             [0, 0, 0, 0,0, 0],
             [0, 0, 0, 0,0, 0],
             [0, 0, 0, 0,0, 0]]
    for i in file.readlines():
        tlist = i.strip().split(' ')
        res = tlist[1]
        dst = tlist[-1]
        img = tlist[0]
        llist[int(res)][int(dst)] += 1
        pa1 = os.path.join(path2, img)
        pa2 = os.path.join(path3, res, dst, img)
        shutil.copy(pa1, pa2)

    path4 = path3+'.txt'
    file3 = open(path4, 'a')
    for k in range(6):
        file3.write(str(k) + ':\t')
        for p in range(6):
            file3.write('[{}]: '.format(p) + '{}'.format(llist[k][p]).zfill(3) + '\t')
        file3.write(' \n')
    file3.close()