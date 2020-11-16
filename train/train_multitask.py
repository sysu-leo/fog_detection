from my_model.MLT_TASK import MulTask
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

#记录训练数据

file = open(cp.get(section, 'acc_multitask_train'), 'w')
file2 = open(cp.get(section, 'acc_multitask_valid'), 'w')

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

# load_model and set gpu_device
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")
torch.cuda.set_device(device)


# load weight
weight_path = '../Parameters/mul_task/epoch20.pth'
model = MulTask()
model = model.to(device)
model.load_state_dict(torch.load(weight_path))
#summary(model.cuda(),  ((4, 448, 448), (3, 488, 488)))
print(model)


# set loss, optimizer, scheduler
loss_cls = nn.CrossEntropyLoss()
loss_pre = nn.SmoothL1Loss()
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
epoch = cp.getint(section, 'epoch')
batchsize = cp.getint(section, 'batchsize')
for i in range(0, epoch):
    running_loss = 0.0
    running_corrects = 0
    running_loss2 = 0.0
    step = 0
    all = int(trainset_size / batchsize +1)
    for _, x1, x2,labels1, labels2 in train_loader1 :
        labels1 = labels1.to(device)
        labels2 = labels2.to(device)
        x1 = x1.to(device)
        x2 = x2.to(device)
        optimizer.zero_grad()
        output1, output2 = model(x1, x2)
        _, pred1 = torch.max(output1, 1)
        est1 = torch.squeeze(output2)

        loss1 = loss_cls(output1, labels1)
        labels2 = labels2.float()
        loss2 = loss_pre(est1, labels2 )
        # losses = loss2*10 + loss1
        losses = loss2
        losses.backward()
        optimizer.step()
        running_loss += loss1.item() * x1.size(0)
        running_loss2 += loss2.item() * x1.size(0)
        running_corrects += torch.sum(pred1 == labels1.data).item()
        print('epoch{}: {}/{} Loss:{:.4f}  ACC:{:.4f}'.format(
            i, step, all,
            losses.item(),
            torch.sum(pred1 == labels1.data).item()/ x1.size(0))
        )
        step+=1
    epoch_loss1 = running_loss / trainset_size
    epoch_loss2 = running_loss2 / trainset_size
    epoch_acc = running_corrects / trainset_size
    print('**************************epoch{} Loss: {:.4f}, {:.4f} Acc: {:.4f}'.format(i, epoch_loss1,epoch_loss2, epoch_acc))
    file.write('{} {:.4f} {:.4f} {:.4f}\n'.format(i, epoch_loss1, epoch_loss2, epoch_acc))
    # valid
    loss_val = 0.0
    correct_val = 0
    for _, inputs1, inputs2,  labels1_1, labels1_2 in valid_loader:
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        labels1_1 = labels1_1.to(device)
        labels1_2 = labels1_2.to(device)
        labels1_2 = labels1_2.float()
        optimizer.zero_grad()
        output1, output2 = model(inputs1, inputs2)
        # print(output1.size())
        _, pred1 = torch.max(output1, 1)
        # print(pred1.size())
        losses1_1 = loss_cls(output1, labels1_1)
        losses1_2 = loss_pre(torch.squeeze(output2), labels1_2)
        loss_val += losses1_1.item() * inputs1.size(0) + losses1_2.item() * inputs1.size(0)
        correct_val += torch.sum(pred1 == labels1_1.data).item()
    val_loss = loss_val / validset_size
    val_acc = correct_val / validset_size
    file2.write('{} {:.4f} {:.4f}\n'.format(i, val_loss, val_acc))
    print('**************************Valid{} Loss: {:.4f} Acc: {:.4f}'.format(i, val_loss, val_acc))




    if i%5 == 0:
        path = '../Parameters/mul_task2/'+'epoch{}'.format(i) + '.pth'
        torch.save(model.state_dict(), path)



