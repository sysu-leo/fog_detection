from my_model.MultiTask import Multi_Task
from Dataset.myDataSet import MyDataSet

import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
import torch.optim
import os
from configparser import ConfigParser

# read parameters
cp = ConfigParser()
cp.read("./param.cfg")
section =cp.sections()[0]
anchors = cp.get(section, "Anchors")
anchors_list = [int(i) for i in anchors.strip().split(',')]
epoch = cp.getint(section, 'epoch')
batchsize = cp.getint(section, 'batchsize')

#记录训练数据

file = ''
epoch = ''
result_file = open('../test/result_' + file +'_' + epoch + '.txt', 'w')
statistics_file = open('../test/statistics_' + file +'_' + epoch + '.txt', 'w')

#read data

simple_transform = transforms.Compose(
    [transforms.Resize((448, 448)),
     transforms.RandomRotation(20),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)

testset = MyDataSet(
    root=cp.get(section, 'root'),
    datatxt=cp.get(section, 'test'),
    tranform=simple_transform
)

# set gpu_device
device = torch.device("cuda:1")
torch.cuda.set_device(device)


# load weight
weight_path = '../Parameters/' + file + '/epoch_' + epoch + '.pth'
model = Multi_Task()
model = model.to(device)
model.load_state_dict(torch.load(weight_path))

test_loader = data.DataLoader(
    dataset=testset,
    batch_size = cp.getint(section, 'batchsize'),
    shuffle=True)


testset_size = len(testset)

res_dict = {0 : [0, 0, 0, 0, 0], 1:[0, 0, 0, 0, 0], 2:[0, 0, 0, 0, 0], 3:[0,0,0,0,0], 4:[0,0,0,0,0]}
average_error = 0.0
# test
for img_names, x1, x2, labels_cls, labels_reg in test_loader:
    #data input
    labels_cls = labels_cls
    labels_reg = labels_reg
    x1 = x1.to(device)
    x2 = x2.to(device)

    cls_out, pre_out = model(x1, x2)
    cls_out = cls_out.to('cpu')
    pre_out = pre_out.to('cpu')
    _, pred = torch.max(cls_out, 1)

    

    for i in range(batchsize):
        img_name = img_names[i]
        level = int(labels_cls[i])
        vis = labels_reg[i]
        pre_level = int(cls_out[i])
        pre_vis = pre_out[i]
        result_file.write(img_name+ ' ' + str(level) + ' ' + str(pre_level) + ' ' + str(vis) + ' ' + str(pre_vis) + '\n')
        res_dict[level][pre_level] += 1
        average_error += abs((pre_vis -  vis))
sum1= []
for i in range(5):
    t_sum = 0
    for j in range(5):
        t_sum += res_dict[j][i]
    sum1.append(t_sum)

sum2 = []
for i in range(5):
    t_sum = 0
    for j in range(5):
        t_sum += res_dict[i][j]
    sum2.append(t_sum)


line_1 = str('{}\t{}\t{}\t{}\t{}\t{}\n'.format(res_dict[0][0], res_dict[0][1], res_dict[0][2], res_dict[0][3], res_dict[0][4], sum2[0]))
line_2 = str('{}\t{}\t{}\t{}\t{}\t{}\n'.format(res_dict[1][0], res_dict[1][1], res_dict[1][2], res_dict[1][3], res_dict[1][4], sum2[1]))
line_3 = str('{}\t{}\t{}\t{}\t{}\t{}\n'.format(res_dict[2][0], res_dict[2][1], res_dict[2][2], res_dict[2][3], res_dict[2][4], sum2[2]))
line_4 = str('{}\t{}\t{}\t{}\t{}\t{}\n'.format(res_dict[3][0], res_dict[3][1], res_dict[3][2], res_dict[3][3], res_dict[3][4], sum2[3]))
line_5 = str('{}\t{}\t{}\t{}\t{}\t{}\n'.format(res_dict[4][0], res_dict[4][1], res_dict[4][2], res_dict[4][3], res_dict[4][4], sum2[4]))
line_6 = str('{}\t{}\t{}\t{}\t{}\nprecision:\n'.format(sum1[0], sum1[1], sum1[2], sum1[3], sum1[4]))

line_7 = str('{}\t{}\t{}\t{}\t{}\n'.format(res_dict[0][0]/float(sum1[0]), res_dict[0][1]/float(sum1[1]), res_dict[0][2]/float(sum1[2]), res_dict[0][3]/float(sum1[3]), res_dict[0][4]/float(sum1[4])))
line_8 = str('{}\t{}\t{}\t{}\t{}\n'.format(res_dict[1][0]/float(sum1[0]), res_dict[1][1]/float(sum1[1]), res_dict[1][2]/float(sum1[2]), res_dict[1][3]/float(sum1[3]), res_dict[1][4]/float(sum1[4])))
line_9 = str('{}\t{}\t{}\t{}\t{}\n'.format(res_dict[2][0]/float(sum1[0]), res_dict[2][1]/float(sum1[1]), res_dict[2][2]/float(sum1[2]), res_dict[2][3]/float(sum1[3]), res_dict[2][4]/float(sum1[4])))
line_10 = str('{}\t{}\t{}\t{}\t{}\n'.format(res_dict[3][0]/float(sum1[0]), res_dict[3][1]/float(sum1[1]), res_dict[3][2]/float(sum1[2]), res_dict[3][3]/float(sum1[3]), res_dict[3][4]/float(sum1[4])))
line_11 = str('{}\t{}\t{}\t{}\t{}\nrecall:\n'.format(res_dict[4][0]/float(sum1[0]), res_dict[4][1]/float(sum1[1]), res_dict[4][2]/float(sum1[2]), res_dict[4][3]/float(sum1[3]), res_dict[4][4]/float(sum1[4])))

line_12 = str('{}\t{}\t{}\t{}\t{}\n'.format(res_dict[0][0]/float(sum2[0]), res_dict[0][1]/float(sum2[0]), res_dict[0][2]/float(sum2[0]), res_dict[0][3]/float(sum2[0]), res_dict[0][4]/float(sum2[0])))
line_13 = str('{}\t{}\t{}\t{}\t{}\n'.format(res_dict[1][0]/float(sum2[1]), res_dict[1][1]/float(sum2[1]), res_dict[1][2]/float(sum2[1]), res_dict[1][3]/float(sum2[1]), res_dict[1][4]/float(sum2[1])))
line_14 = str('{}\t{}\t{}\t{}\t{}\n'.format(res_dict[2][0]/float(sum2[2]), res_dict[2][1]/float(sum2[2]), res_dict[2][2]/float(sum2[2]), res_dict[2][3]/float(sum2[2]), res_dict[2][4]/float(sum2[2])))
line_15 = str('{}\t{}\t{}\t{}\t{}\n'.format(res_dict[3][0]/float(sum2[3]), res_dict[3][1]/float(sum2[3]), res_dict[3][2]/float(sum2[3]), res_dict[3][3]/float(sum2[3]), res_dict[3][4]/float(sum2[3])))
line_16 = str('{}\t{}\t{}\t{}\t{}\n'.format(res_dict[4][0]/float(sum2[4]), res_dict[4][1]/float(sum2[3]), res_dict[4][2]/float(sum2[3]), res_dict[4][3]/float(sum2[3]), res_dict[4][4]/float(sum2[3])))

line_17 = str('average precision : {}\n'.format((res_dict[0][0]/float(sum1[0]) + res_dict[1][1]/float(sum1[1]) + res_dict[2][2]/float(sum1[2]) + res_dict[3][3]/float(sum1[3]) + res_dict[4][4]/float(sum1[4]))/ 5.0))
line_18 = str('average recal : {}\n'.format((res_dict[0][0]/float(sum2[0]) + res_dict[1][1]/float(sum2[1]) + res_dict[2][2]/float(sum2[2]) + res_dict[3][3]/float(sum2[3]) + res_dict[4][4]/float(sum2[4]))/ 5.0))
line_19 = str('average error : {}\n'.format(average_error /  testset_size))

statistics_file.write(line_1)
statistics_file.write(line_2)
statistics_file.write(line_3)
statistics_file.write(line_4)
statistics_file.write(line_5)
statistics_file.write(line_6)
statistics_file.write(line_7)
statistics_file.write(line_8)
statistics_file.write(line_9)
statistics_file.write(line_10)
statistics_file.write(line_11)
statistics_file.write(line_12)
statistics_file.write(line_13)
statistics_file.write(line_14)
statistics_file.write(line_15)
statistics_file.write(line_16)
statistics_file.write(line_19)
statistics_file.write(line_18)
statistics_file.write(line_19)

result_file.close()
statistics_file.close()

