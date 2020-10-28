import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import DataProcess as dp
import torch.utils.model_zoo as model_zoo
import cv2
import numpy as np
from AlexNet import LRN


'''rgbd_Net
four channel network
intput : 448*448*4
output : 1024
keypoint:
(1)hole convolution
(2)Global Average pooling
(3)channel fusion
'''
device = torch.device("cuda:0")
class rgbd_Net(nn.Module):
    def __init__(self):
        super(rgbd_Net, self).__init__()
        self.FeatureMap = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=96, kernel_size=5, stride=4, padding=4,dilation=2),
            nn.BatchNorm2d(96),
            nn.ELU(inplace=True),#112*112
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=2, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),#56*56
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #28*28
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True),#14*14
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) #7*7
        )
        self.linear = nn.Linear(in_features=7*7*384, out_features=1024)
    def forward(self, x):
        x = self.FeatureMap(x)
        x= x.view(x.size(0), 7*7*384)
        x = self.linear(x)
        return x

'''sliceNet
input: 224*224
output:1024
keytpoint:
(1)each piece was treated separately
'''

class sliceNet(nn.Module):
    def __init__(self, input_channel = 3):
        super(sliceNet, self).__init__()
        self.input_channel = input_channel
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = self.input_channel, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 224* 224*64
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 112*112*96
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #56*56*128
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #28*28*256
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #7*7*384
        self.linear = nn.Sequential(
            nn.Linear(in_features=7*7*384, out_features=1024),
            nn.ELU(inplace=True)
        )
        self.FeatrueMap = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        )

    def forward(self, x):
        x = self.FeatrueMap(x)
        x = x.view(x.size(0), 7*7*384)
        x = self.linear(x)
        return x
'''rgbd_Net
four channel network
intput : 448*448*3
output : 1024
keypoint:
(1)hole convolution
(2)Global Average pooling
(3)channel fusion
'''
class rgbd_Net1(nn.Module):
    def __init__(self):
        super(rgbd_Net1, self).__init__()
        self.FeatureMap = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=96, kernel_size=5, stride=4, padding=4, dilation=2),
            nn.BatchNorm2d(96),
            nn.ELU(inplace=True),  # 112*112
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=2, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),  # 56*56
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28*28
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True),  # 14*14
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 7*7
        )
        self.convFc = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ELU(inplace=True),  # 14*14
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3)  # 1*1*1024
        )

    def forward(self, x):
        x = self.FeatureMap(x)
        x = self.convFc(x)
        x = x.view(x.size(0), 1024)
        return x

'''sliceNet
input: 112*112*3
output:1024
keytpoint:
(1)each piece was treated separately
'''
class sliceNet1(nn.Module):
    def __init__(self, input_channel=3):
        super(sliceNet1, self).__init__()
        self.input_channel = input_channel
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=64, kernel_size=3, padding=1),  # 112*112*96
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 112* 112 *64
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 112*112*96
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 28*28*128
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 7*7*256

        self.convFc = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3)
        )
        # 1*1*1024
        self.FeatrueMap = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3
        )

    def forward(self, x):
        x = self.FeatrueMap(x)
        x = self.convFc(x)
        x = x.view(x.size(0), 1024)
        return x


'''classifier Net
input:input_channel * wight*height
output:6classes
keyPoints:
(1)GAP
'''
class claasifierNet(nn.Module):
    def __init__(self, input_channel):
        super(claasifierNet, self).__init__()
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
        x = F.softmax(x)
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
        # x = self.avgpool(F.relu(self.conv2(x)))
        x = F.softmax(x)
        return x



'''
使用GAP的RGBD单特征网络结构rgbdNet
'''

class testNet1(nn.Module):
    def __init__(self):
        super(testNet1, self).__init__()
        self.rgbd = rgbd_Net()
        self.classifier = claasifierNet(input_channel=1)

    def forward(self, x):
        x = self.rgbd(x)
        x = torch.reshape(x, (x.size(0), 1, 32, 32))
        x = self.classifier(x)
        return x

'''
使用GAP的slice单特征网络结构sliceNet
'''
class testNet2(nn.Module):
    def __init__(self):
        super(testNet2, self).__init__()
        self.slice = sliceNet(input_channel=3)
        self.classifier = claasifierNet(input_channel=1)

    def forward(self, x):
        x = self.slice(x)
        x = torch.reshape(x, (x.size(0), 1, 32, 32))
        x = self.classifier(x)
        return x


'''
使用GAP的RGBD单特征网络结构rgbdNet1
'''
class testNet3(nn.Module):
    def __init__(self):
        super(testNet3, self).__init__()
        self.rgbd = rgbd_Net1()
        self.classifier = claasifierNet(input_channel=1)

    def forward(self, x):
        x = self.rgbd(x)
        x = torch.reshape(x, (x.size(0), 1, 32, 32))
        x = self.classifier(x)
        return x


'''
使用GAP的slice单特征网络结构sliceNet1
'''
class testNet4(nn.Module):
    def __init__(self):
        super(testNet4, self).__init__()
        self.slice = sliceNet1(input_channel=18)
        self.classifier = claasifierNet(input_channel=1)

    def forward(self, x):
        x = self.slice(x)
        x = torch.reshape(x, (x.size(0), 1, 32, 32))
        x = self.classifier(x)
        return x
'''
整体结构测试1
'''
class testNet5(nn.Module):
    def __init__(self):
        super(testNet5, self).__init__()
        self.slice = sliceNet()
        self.rgbd = rgbd_Net()
        self.classifier = claasifierNet(input_channel=2)

    def forward(self, x1, x2):
        x1 = self.rgbd(x1)
        x1 = torch.reshape(x1, (x1.size(0), 1, 32, 32))
        x2 = self.slice(x2)
        x2 = torch.reshape(x2, (x2.size(0), 1, 32, 32))
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x
'''
整体结构测试2
'''
class testNet6(nn.Module):
    def __init__(self):
        super(testNet6, self).__init__()
        self.slice = sliceNet1()
        self.rgbd = rgbd_Net1()
        self.classifier = claasifierNet(input_channel=2)

    def forward(self, x1, x2):
        x1 = self.rgbd(x1)
        x1 = torch.reshape(x1, (x1.size(0), 1, 32, 32))
        x2 = self.slice(x2)
        x2 = torch.reshape(x2, (x2.size(0), 1, 32, 32))
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x




