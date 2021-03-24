from my_model.DisEstimation_Single_Nocas import  DisEstimation_Single
from my_model.myloss import Relative_loss
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

file_train = open(cp.get(section, 'acc_vis_train_nocas'), 'w')
file_valid = open(cp.get(section, 'acc_vis_valid_nocas'), 'w')

#read data

simple_transform = transforms.Compose(
    [transforms.Resize((448, 448)),
     transforms.RandomRotation(20),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)

trainset = MyDataSet(
    root=cp.get(section, 'root'),
    datatxt=cp.get(section, 'train'),
    tranform=simple_transform
)
validset = MyDataSet(
    root=cp.get(section, 'root'),
    datatxt=cp.get(section, 'valid'),
    tranform=simple_transform
)

# set gpu_device
device = torch.device("cuda:1")
torch.cuda.set_device(device)


# load weight
#weight_path = '../Parameters/mul_task2/epoch_fea_20.pth'
model = DisEstimation_Single()
model = model.to(device)
# fea_model.load_state_dict(torch.load(weight_path))
#summary(model.cuda(),  ((4, 448, 448), (3, 488, 488)))
# print(model)


# set loss_function

loss_pre = nn.L1Loss()

# set optimizer
optimizer_pre = torch.optim.SGD(
    model.parameters(),
    lr=cp.getfloat(section, 'lr2'),
    momentum=cp.getfloat(section,'momentum'),
    weight_decay=cp.getfloat(section, 'weight_decay')
)

# set schecual
scheduler1 = torch.optim.lr_scheduler.StepLR(
    optimizer_pre,
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
for i in range(0, epoch):
    running_loss = 0.0
    running_corrects = 0
    running_loss2 = 0.0
    step = 0
    all = int(trainset_size / batchsize +1)
    for _, x1, x2, labels_cls, labels_reg in train_loader1:
        #data input
        labels_cls = labels_cls.to(device)
        labels_reg = labels_reg.to(device)
        x1 = x1.to(device)
        x2 = x2.to(device)
        #data output
        optimizer_pre.zero_grad()
        pre_out = model(x1, x2)
        #loss calculation
        labels_reg = labels_reg.float()
        losses_pre = loss_pre(pre_out, labels_reg)
        losses_pre.backward()
        optimizer_pre.step()

        running_loss += losses_pre.item() * x1.size(0)

        print('epoch{}: {}/{} Loss:{:.4f}'.format(
            i, step, all,
            losses_pre.item())
        )

        #release cache
        labels_cls = labels_cls.to('cpu')
        labels_reg = labels_reg.to('cpu')
        x1 = x1.to('cpu')
        x2 = x2.to('cpu')
        pre_out = pre_out.to('cpu')
        torch.cuda.empty_cache()
        step += 1

    epoch_loss_pre = running_loss / trainset_size
    print('**************************epoch{} Loss: {:.4f}'.format(i, epoch_loss_pre))
    file_train.write('{} {:.4f}\n'.format(i, epoch_loss_pre))

    # valid
    loss_val = 0.0
    correct_val = 0
    for _, inputs1, inputs2,  labels1_cls, labels1_reg in valid_loader:
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        labels1_cls = labels1_cls.to(device)
        labels1_reg = labels1_reg.to(device)
        optimizer_pre.zero_grad()
        labels1_reg = labels1_reg.float()
        output_pre = model(inputs1, inputs2)
        losses1_pre = loss_pre(output_pre, labels1_reg)
        loss_val += losses1_pre.item() * inputs1.size(0)

        # release cache
        labels1_cls = labels1_cls.to('cpu')
        labels2_reg = labels1_reg.to('cpu')
        inputs1 = inputs1.to('cpu')
        inputs2 = inputs2.to('cpu')
        output_pre = output_pre.to('cpu')
        torch.cuda.empty_cache()
    val_loss = loss_val / validset_size

    print('**************************Valid{} Loss: {:.4f}'.format(i, val_loss))
    file_valid.write('{} {:.4f}\n'.format(i, val_loss))


    if i%10 == 0:
        path = '/home/dell/Documents/Parameters/vis_single_Nocas/'+'epoch_{}'.format(i) + '.pth'
        torch.save(model.state_dict(), path)



