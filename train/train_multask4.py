from my_model.MLT_TASK import MulTask
from Dataset.myDataSet import MyDataSet
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
import torch.optim
from my_model.classifierNet import claasifierNet1, mlp3
from my_model.MLT_TASK import FeaModel

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

file = open(cp.get(section, 'acc_multitask_train4'), 'w')
file2 = open(cp.get(section, 'acc_multitask_valid4'), 'w')

#read data

simple_transform = transforms.Compose(
    [transforms.Resize((448, 448)),
     transforms.RandomRotation(20),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)

trainset = MyDataSet(
    root=cp.get(section, 'root'),
    file_rgb = cp.get(section, 'RGD_data'),
    file_d = cp.get(section, 'dark_data'),
    file_slice=cp.get(section, 'slice_data'),
    datatxt=cp.get(section, 'train'),
    tranform=simple_transform
)
validset = MyDataSet(
    root=cp.get(section, 'root'),
    file_rgb=cp.get(section, 'RGD_data'),
    file_d=cp.get(section, 'dark_data'),
    file_slice=cp.get(section, 'slice_data'),
    datatxt=cp.get(section, 'valid'),
    tranform=simple_transform
)

# set gpu_device
device = torch.device("cuda:0")
torch.cuda.set_device(device)


# load weight
weight_path = '../Parameters/mul_task2/epoch_fea_20.pth'
fea_model = FeaModel()
#cls_model = claasifierNet1(input_channel=2)
pre_model = mlp3()
fea_model = fea_model.to(device)
pre_model = pre_model.to(device)
fea_model.load_state_dict(torch.load(weight_path))
#summary(model.cuda(),  ((4, 448, 448), (3, 488, 488)))
# print(model)


# set loss_function
loss_cls = nn.CrossEntropyLoss()
loss_pre = nn.SmoothL1Loss()


# set optimizer
optimizer1 = torch.optim.SGD(
    fea_model.parameters(),
    lr=cp.getfloat(section, 'lr'),
    momentum=cp.getfloat(section,'momentum'),
    weight_decay=cp.getfloat(section, 'weight_decay')
)

optimizer2 = torch.optim.SGD(
    pre_model.parameters(),
    lr=cp.getfloat(section, 'lr2'),
    momentum=cp.getfloat(section,'momentum'),
    weight_decay=cp.getfloat(section, 'weight_decay')
)

# set schecual
scheduler1 = torch.optim.lr_scheduler.StepLR(
    optimizer1,
    step_size =cp.getint(section, 'step_size'),
    gamma = cp.getfloat(section, 'gamma')
)

scheduler2 = torch.optim.lr_scheduler.StepLR(
    optimizer2,
    step_size =cp.getint(section, 'step_size'),
    gamma = cp.getfloat(section, 'gamma')
)


classifier = nn.Sequential(nn.BatchNorm2d(2),claasifierNet1(input_channel=2))

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
fea_model.train()
pre_model.train()
for i in range(0, epoch):
    running_loss = 0.0
    running_corrects = 0
    running_loss2 = 0.0
    step = 0
    all = int(trainset_size / batchsize +1)
    for _, x1, x2, labels1, labels2 in train_loader1 :

        labels1 = labels1.to(device)
        labels2 = labels2.to(device)
        x1 = x1.to(device)
        x2 = x2.to(device)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        #feature exactor finished: output1 & output2 is the exact feature.
        output1, output2 = fea_model(x1, x2)

        # feature mid process
        x_1 = torch.reshape(output1, (output1.size(0), 1, 32, 32))
        x_2 = torch.reshape(output1, (output2.size(0), 1, 32, 32))
        x = torch.cat((x_1, x_2), dim = 1)

        #cls_model or est_model
        output = pre_model(x)


        # _, pred1 = torch.max(output, 1)

        # est1 = torch.squeeze(output2)
        labels2 = labels2.float()
        loss = loss_pre(output, labels2)
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        running_loss += loss.item() * x1.size(0)
        # running_loss2 += loss2.item() * x1.size(0)
        # running_corrects += torch.sum(pred1 == labels1.data).item()
        # print('epoch{}: {}/{} Loss:{:.4f}  ACC:{:.4f}'.format(
        #     i, step, all,
        #     loss.item(),
        #     torch.sum(pred1 == labels1.data).item()/ x1.size(0))
        # )
        print('epoch{}: {}/{} Loss:{:.4f}'.format(
                i, step, all,
                loss.item())
            )
        step+=1

        #release cache
        labels1 = labels1.to('cpu')
        labels2 = labels2.to('cpu')
        x1 = x1.to('cpu')
        x2 = x2.to('cpu')
        output1 = output1.to('cpu')
        output2 = output2.to('cpu')
        x = x.to('cpu')
        output = output.to('cpu')
        torch.cuda.empty_cache()
        #-----------------#
    epoch_loss1 = running_loss / trainset_size
    # epoch_loss2 = running_loss2 / trainset_size
    #epoch_acc = running_corrects / trainset_size
    epoch_acc = 0.0
    print('**************************epoch{} Loss: {:.4f},Acc: {:.4f}'.format(i, epoch_loss1, epoch_acc))
    file.write('{} {:.4f} {:.4f}\n'.format(i, epoch_loss1, epoch_acc))
    # valid
    loss_val = 0.0
    correct_val = 0
    for _, inputs1, inputs2,  labels1_1, labels1_2 in valid_loader:
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        labels1_1 = labels1_1.to(device)
        labels1_2 = labels1_2.to(device)
        # labels1_2 = labels1_2.float()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        output1_, output2_ = fea_model(inputs1, inputs2)
        x_1_ = torch.reshape(output1_, (output1_.size(0), 1, 32, 32))
        x_2_ = torch.reshape(output1_, (output2_.size(0), 1, 32, 32))
        x_ = torch.cat((x_1_, x_2_), dim=1)
        labels1_2 = labels1_2.float()
        output_ = pre_model(x_)
        # _, pred1 = torch.max(output_, 1)
        # print(pred1.size())
        # losses1_1 = loss_cls(output_, labels1_1)
        losses1_1 = loss_pre(output_, labels1_2)
        # losses1_2 = loss_pre(torch.squeeze(output2), labels1_2)
        loss_val += losses1_1.item() * inputs1.size(0)
        # loss_val += losses1_1.item() * inputs1.size(0) + losses1_2.item() * inputs1.size(0)
        # correct_val += torch.sum(pred1 == labels1_1.data).item()
        correct_val = 0.0
        # release cache
        labels1_1 = labels1_1.to('cpu')
        labels2_1 = labels1_2.to('cpu')
        inputs1 = inputs1.to('cpu')
        inputs2 = inputs2.to('cpu')
        output_ = output_.to('cpu')
        torch.cuda.empty_cache()
        # -----------------#
    val_loss = loss_val / validset_size
    val_acc = correct_val / validset_size
    file2.write('{} {:.4f} {:.4f}\n'.format(i, val_loss, val_acc))
    print('**************************Valid{} Loss: {:.4f} Acc: {:.4f}'.format(i, val_loss, val_acc))




    if i%5 == 0:
        path = '../Parameters/mul_task4/'+'epoch_fea_{}'.format(i) + '.pth'
        torch.save(fea_model.state_dict(), path)
        path1 = '../Parameters/mul_task4/' + 'epoch_pre_{}'.format(i) + '.pth'
        torch.save(pre_model.state_dict(), path1)



