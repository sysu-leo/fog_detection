
from my_model.sliceNet1 import testsliceNet1
from Dataset.MyDataSet import MyDataSet
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import os
from torchsummary import summary
from configparser import ConfigParser
import numpy as np

# read parameters

cp = ConfigParser()
cp.read("./param.cfg")
section =cp.sections()[0]

#记录训练数据

file = open(cp.get(section, 'acc_slice1_train'), 'w')
file2 = open(cp.get(section, 'acc_slice1_valid'), 'w')

#read data

simple_transform = transforms.Compose(
    [transforms.Resize((448, 448)),
     transforms.RandomRotation(20),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)

trainset = MyDataSet(
    root=cp.get(section, 'root'),
    file_rgbd = cp.get(section, 'RGD_data'),
    file_dark = cp.get(section, 'dark_data'),
    file_slice='slice',
    datatxt=cp.get(section, 'train'),
    tranform=simple_transform,
    is_slice=True
)
validset = MyDataSet(
    root=cp.get(section, 'root'),
    file_rgbd=cp.get(section, 'RGD_data'),
    file_dark=cp.get(section, 'dark_data'),
    file_slice='slice',
    datatxt=cp.get(section, 'valid'),
    tranform=simple_transform,
    is_slice=True
)



# load_model and set gpu_device
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:1")
torch.cuda.set_device(device)


# load weight
# model.load_state_dict(torch.load(weight_path))
model = testsliceNet1()
model = model.to(device)
summary(model.cuda(),  (3, 448,448))


# set loss, optimizer, scheduler
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=cp.getfloat(section, 'lr'),
    momentum=cp.getfloat(section,'momentum'),
    weight_decay=cp.getfloat(section, 'weight_decay')
)


scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size =cp.getint(section, 'step_size'),
    gamma = cp.getfloat(section, 'gamma')
)

# load data
train_loader1 = data.DataLoader(
    dataset=trainset,
    batch_size = cp.getint(section, 'batchsize'),
    shuffle=True)

valid_loader = data.DataLoader(
    dataset=validset,
    batch_size=10)


trainset_size = len(trainset)
validset_size = len(validset)

# train
model.train()
model.train()
epoch = cp.getint(section, 'epoch')
batchsize = cp.getint(section, 'batchsize')
for i in range(epoch):
    running_loss = 0.0
    running_corrects = 0
    step = 0
    all = int(trainset_size / batchsize + 1)
    for _, x, labels in train_loader1:
        labels = labels.to(device)
        x = x.to(device)
        optimizer.zero_grad()
        output = model(x)
        _, pred = torch.max(output, 1)
        losses = loss(output, labels)
        losses.backward()
        optimizer.step()
        running_loss += losses.item() * x.size(0)
        running_corrects += torch.sum(pred == labels.data).item()
        print('epoch{}: {}/{} Loss:{:.4f}  ACC:{:.4f}'.format(
            i, step, all,
            losses.item(),
            torch.sum(pred == labels.data).item() / x.size(0))
        )
        step += 1
    epoch_loss = running_loss / trainset_size
    epoch_acc = running_corrects / trainset_size
    print('**************************epoch{} Loss: {:.4f} Acc: {:.4f}'.format(i, epoch_loss, epoch_acc))
    file.write('{} {:.4f} {:.4f}\n'.format(i, epoch_loss, epoch_acc))

    # valid
    loss_val = 0.0
    correct_val = 0
    for _, inputs1, labels1 in valid_loader:
        inputs1 = inputs1.to(device)
        labels1 = labels1.to(device)

        optimizer.zero_grad()
        output = model(inputs1)
        _, pred = torch.max(output, 1)

        losses = loss(output, labels1)

        loss_val += losses.item() * inputs1.size(0)
        correct_val += torch.sum(pred == labels1.data).item()
    val_loss = loss_val / validset_size
    val_acc = correct_val / validset_size
    file2.write('{} {:.4f} {:.4f}\n'.format(i, val_loss, val_acc))
    print('**************************Valid{} Loss: {:.4f} Acc: {:.4f}'.format(i, val_loss, val_acc))

    if i % 5 == 0:
        path = '../Parameters/model_slice/' + 'epoch{}'.format(i) + '.pth'
        torch.save(model.state_dict(), path)



