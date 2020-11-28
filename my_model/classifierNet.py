import torch
import torch.nn as nn
import torch.nn.functional as F


'''classifier Net1
input:input_channel * wight*height
output:6classes
keyPoints:
(1)GAP
'''
device = torch.device("cuda:0")



class claasifierNet1(nn.Module):
    def __init__(self, input_channel):
        super(claasifierNet1, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size= 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=6, kernel_size=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.classifier(x)
        x = self.classifier2(x)
        x = x.view(x.size(0), 1*1*6)
        #x = self.softmax(x)
        return x


'''classifier Net2
input:input_channel * wight*height
output:6classes
keyPoints:
(1)FC
'''


class claasifierNet2(nn.Module):
    def __init__(self, input_channel, output_channel = 6):
        super(claasifierNet2, self).__init__()
        self.softmax = nn.Softmax(dim=6)
        self.conv1 = nn.Conv2d(in_channels = input_channel, out_channels=128, kernel_size=3, stride = 2, padding=1) #32*32*6
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels=128, kernel_size=3, stride = 1, padding=1)#
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=6, kernel_size=3, stride = 1)
        self.linear = nn.Linear(in_features=8*8*6, out_features=6)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x))) #8*8*96
        x = F.relu(self.conv3(x)) #8*8*256
        x = x.view(x.size(0), 1 * 1 * 6)
        x = self.linear(F.relu(self.conv2(x)).view(x.size(0), 8*8*6))
        x = self.softmax(x)
        return x


class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=4)
        self.linear1 = nn.Linear(in_features=4, out_features=64)
        self.linear2 = nn.Linear(in_features = 64, out_features = 64)
        self.linear3 = nn.Linear(in_features=64, out_features = 1)
    def forward(self, x1, x2, x3):
        x_1, _ = torch.max(x1,dim=1, keepdim=True)
        x_2, _= torch.max(x2, dim = 1, keepdim=True)

        x_3 = torch.mean(x1, dim=1, keepdim=True)

        x_4 = torch.mean(x2, dim=1, keepdim=True)

        x = torch.cat((x_1, x_2, x_3, x_4), dim=1)

        x = self.softmax(x)
        x = (F.relu(self.dropout(self.linear1(x))))
        x = (F.relu(self.dropout(self.linear2(x))))
        x = (F.relu(self.dropout(self.linear3(x))))
        return x

class mlp2(nn.Module):
    def __init__(self):
        super(mlp2, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=4)
        self.linear1 = nn.Linear(in_features=6, out_features=64)
        self.linear2 = nn.Linear(in_features = 64, out_features = 64)
        self.linear3 = nn.Linear(in_features=64, out_features = 1)
    def forward(self, x1, x2, x3):
        x_1, _ = torch.max(x1,dim=1, keepdim=True)

        x_2, _ = torch.max(x2, dim = 1, keepdim=True)

        x_3 = torch.mean(x1, dim=1, keepdim=True)

        x_4 = torch.mean(x2, dim=1, keepdim=True)

        x_5, _ = torch.min(x1, dim=1, keepdim=True)

        x_6, _ = torch.min(x2, dim = 1, keepdim=True)


        x = torch.cat((x_1, x_2, x_3, x_4, x_5, x_6), dim=1)

        x = self.softmax(x)
        x = (F.relu(self.dropout(self.linear1(x))))
        x = (F.relu(self.dropout(self.linear2(x))))
        x = (F.relu(self.dropout(self.linear3(x))))
        return x

class mlp3(nn.Module):
    def __init__(self):
        super(mlp3, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1)
        self.linear1 = nn.Linear(in_features=2048, out_features=1024)
        self.linear2 = nn.Linear(in_features = 1024, out_features = 1024)
        self.linear3 = nn.Linear(in_features=1024, out_features=256)
        self.linear4 = nn.Linear(in_features=256, out_features = 6)
    def forward(self, x1, x2):
        k = x1.size(0)
        print(k)
        x1_ = torch.tensor([35.0/500.0, 75.0/500.0, 150.0/500.0, 250.0/500.0, 350.0/500.0, 1.0])
        tmp_x = torch.tensor([35.0/500.0, 75.0/500.0, 150.0/500.0, 250.0/500.0, 350.0/500.0, 1.0]).reshape((6, 1)).to(device)
        print(tmp_x.size())
        x = torch.cat((x1, x2), dim=1)
        x = (F.relu(self.dropout(self.linear1(x))))
        x = (F.relu(self.dropout(self.linear2(x))))
        x = (F.relu(self.dropout(self.linear3(x))))
        x = (F.relu(self.dropout(self.linear4(x))))
        x = self.softmax(x)
        print(x.size())
        x = torch.mm(x,tmp_x)
        x = torch.squeeze(x)
        return x

'''classifier Net1
input:input_channel * wight*height
output:6classes
keyPoints:
(1)GAP
'''


class claasifierNet3(nn.Module):
    def __init__(self, input_channel):
        super(claasifierNet3, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size= 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=6, kernel_size=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.classifier(x)
        x = self.classifier2(x)
        tmp_x = torch.tensor([35.0 / 500.0, 75.0 / 500.0, 150.0 / 500.0, 250.0 / 500.0, 350.0 / 500.0, 1.0]).reshape(
            (6, 1)).to(device)
        x = x.view(x.size(0), 1*1*6)
        x = self.softmax(x)
        x = torch.mm(x, tmp_x)
        x = torch.squeeze(x)
        return x



