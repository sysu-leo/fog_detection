from model.AlexNet import AlexNet
from model.VGG16 import Vgg16
from my_model.EnvNet4 import EnvNet4
from my_model.sliceNet1 import testsliceNet1
from my_model.rgbdNet1 import testrgbdNet1
from my_model.sliceNet2 import testsliceNet2
from my_model.rgbdNet2 import testrgbdNet2
#from Dataset.MyDataSet import MyDataSet
from script.getRes import gerRes


from Dataset.myDataSet2 import MyDataSet


import torchvision
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim
import os
from configparser import ConfigParser
from torchsummary import summary

cp = ConfigParser()
cp.read("./param.cfg")
section =cp.sections()[0]


#read data
epoch = 100
simple_transform = transforms.Compose(
    [transforms.Resize((448, 448)),
     transforms.RandomRotation(20),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)

testset = MyDataSet(
    root=cp.get(section, 'root'),
    file_rgb = cp.get(section, 'RGD_data'),
    file_d = cp.get(section, 'dark_data'),
    file_slice=cp.get(section, 'slice_data'),
    datatxt=cp.get(section, 'test'),
    tranform=simple_transform
)

# load_model and set gpu_device
#
device = torch.device("cuda:0")
torch.cuda.set_device(device)


# load weight
weight_path = '../Parameters/model_envnet4/epoch{}.pth'.format(epoch)
model = EnvNet4()
model = model.to(device)
model.load_state_dict(torch.load(weight_path))


# set loss, optimizer, scheduler
loss = nn.CrossEntropyLoss()
# load data
test_loader = data.DataLoader(
    dataset=testset,
    batch_size = cp.getint(section, 'test_batchsize'),
    shuffle=True
)



testset_size = len(testset)

# train
correct = 0
res_path = '../test_result/envNet4_epoch{}.txt'.format(epoch)
res_file = open(res_path, 'w')
correct_0 = 0
correct_1 = 0
correct_2 = 0
correct_3 = 0
correct_4 = 0
correct_5 = 0

pred_0 = 0
pred_1 = 0
pred_2 = 0
pred_3 = 0
pred_4 = 0
pred_5 = 0


running_loss = 0.0
running_corrects = 0
step = 0
count = 0
all = int(testset_size)
for img_name, x1, x2,labels in test_loader:
    x1 = x1.to(device)
    x2 = x2.to(device)
    output = model(x1, x2)
    print(count)
    count += 1
    prob, preds = torch.max(output, 1)
    for i in range(32):
        try:
            label = labels[i]
            pred = preds[i]
            res_file.write(img_name[i] + ' ' + str(label.item()) + ' ' + str(pred.item()) + '\n')
            if label == pred:
                correct += 1
            if pred.item() == 0:
                pred_0 += 1
                if label == pred.item():
                    correct_0 += 1
            elif pred.item() == 1:
                pred_1 += 1
                if label == pred.item():
                    correct_1 += 1
            elif pred.item() == 2:
                pred_2 += 1
                if label == pred.item():
                    correct_2 += 1
            elif pred.item() == 3:
                pred_3 += 1
                if label == pred.item():
                    correct_3 += 1
            elif pred.item() == 4:
                pred_4 += 1
                if label == pred.item():
                    correct_4 += 1
            elif pred.item() == 5:
                pred_5 += 1
                if label == pred.item():
                    correct_5 += 1
        except:
            print('error')
file4 = open('../Result/'+res_path.split('/')[-1], 'w')
file4.write('all_acc: {}\n'.format(correct / all))
file4.write('class0_pred:{}\n'.format(pred_0))
file4.write('class1_pred: {} , class1_pred: {}, per:{},  recall:{}\n'.format(pred_1, correct_1, correct_1 / pred_1,
                                                                     correct_1 / 250.0))
file4.write('class2_pred: {} , class1_pred: {}, per:{},  recall:{}\n'.format(pred_2, correct_2, correct_2 / pred_2,
                                                                     correct_2 / 250.0))
file4.write('class3_pred: {} , class1_pred: {}, per:{},  recall:{}\n'.format(pred_3, correct_3, correct_3 / pred_3,
                                                                     correct_3 / 250.0))
file4.write('class4_pred: {} , class1_pred: {}, per:{},  recall:{}\n'.format(pred_4, correct_4, correct_4 / pred_4,
                                                                     correct_4 / 250.0))
file4.write('class5_pred: {} , class1_pred: {}, per:{},  recall:{}\n'.format(pred_5, correct_5, correct_5 / pred_5,
                                                                     correct_5 / 250.0))
res_file.close()
file4.close()

gerRes(res_path)









