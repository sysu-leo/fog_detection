from my_model.FogLevel_Single import FogLevel_Single
from Dataset.myDataSet2 import MyDataSet

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

file_train = open(cp.get(section, 'acc_fog_level_train'), 'w')
file_valid = open(cp.get(section, 'acc_fog_level_valid'), 'w')

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
#weight_path = '../Parameters/mul_task2/epoch_fea_20.pth'
model = FogLevel_Single()
model = model.to(device)
# fea_model.load_state_dict(torch.load(weight_path))
#summary(model.cuda(),  ((4, 448, 448), (3, 488, 488)))
# print(model)


# set loss_function

loss_cls = nn.CrossEntropyLoss()

# set optimizer
optimizer_cls = torch.optim.SGD(
    fea_model.parameters(),
    lr=cp.getfloat(section, 'lr'),
    momentum=cp.getfloat(section,'momentum'),
    weight_decay=cp.getfloat(section, 'weight_decay')
)

# set schecual
scheduler1 = torch.optim.lr_scheduler.StepLR(
    optimizer1,
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
        optimizer1_cls.zero_grad()
        cls_out = model(x1, x2)
        _, pred = torch.max(cls_out, 1)
        #loss calculation
        losses_cls = loss_cls(cls_out, labels_cls)
        losses_cls.backward()
        optimizer_cls.step()

        running_loss += losses_cls.item() * x1.size(0)
        running_corrects += torch.sum(pred == labels_cls.data).item()

        print('epoch{}: {}/{} Loss:{:.4f}  ACC:{:.4f}'.format(
            i, step, all,
            losses_cls.item(),
            torch.sum(pred == labels_cls.data).item()/ x1.size(0))
        )

        #release cache
        labels_cls = labels_cls.to('cpu')
        labels_reg = labels_reg.to('cpu')
        x1 = x1.to('cpu')
        x2 = x2.to('cpu')
        cls_out = cls_out.to('cpu')
        torch.cuda.empty_cache()
        step += 1

    epoch_loss_cls = running_loss / trainset_size
    epoch_acc = running_corrects / trainset_size
    print('**************************epoch{} Loss: {:.4f},Acc: {:.4f}'.format(i, epoch_loss_cls, epoch_acc))
    file.write('{} {:.4f} {:.4f}\n'.format(i, epoch_loss_cls, epoch_acc))

    # valid
    loss_val = 0.0
    correct_val = 0
    for _, inputs1, inputs2,  labels1_cls, labels1_reg in valid_loader:
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        labels1_cls = labels1_cls.to(device)
        labels1_reg = labels1_reg.to(device)
        # labels1_2 = labels1_2.float()
        optimizer_cls.zero_grad()

        output_cls = model(inputs1, inputs2)
        _, pred1 = torch.max(output_cls, 1)
        losses1_cls = loss_cls(output_cls, labels1_cls)
        loss_val += losses1_cls.item() * inputs1.size(0)
        correct_val += torch.sum(pred1 == labels1_cls.data).item()

        # release cache
        labels1_cls = labels1_cls.to('cpu')
        labels2_reg = labels1_reg.to('cpu')
        inputs1 = inputs1.to('cpu')
        inputs2 = inputs2.to('cpu')
        output_cls = output_cls.to('cpu')
        torch.cuda.empty_cache()
    val_loss = loss_val / validset_size
    val_acc = correct_val / validset_size

    print('**************************Valid{} Loss: {:.4f} Acc: {:.4f}'.format(i, val_loss, val_acc))
    file2.write('{} {:.4f} {:.4f}\n'.format(i, val_loss, val_acc))


    if i%10 == 0:
        path = '../Parameters/mul_task4/'+'epoch_fea_{}'.format(i) + '.pth'
        torch.save(fea_model.state_dict(), path)
        path1 = '../Parameters/mul_task4/' + 'epoch_pre_{}'.format(i) + '.pth'
        torch.save(pre_model.state_dict(), path1)



