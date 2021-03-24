from torch import nn

import os
import sys
import torch
from my_model.TaskModule import FogLevel_Classify, VisDistance_Estimation
from my_model.TaskModule import FogLevel_Classify_FC, Visibility_Estimation_FC

current_dir = os.getcwd()    # obtain work dir
sys.path.append(current_dir) # add work dir to sys path

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.task1 = FogLevel_Classify_FC(input_channel=1024)
        self.task2 = Visibility_Estimation_FC(input_channel=1024)

        # 第一层，2个卷积层和一个最大池化层
        self.layer1 = nn.Sequential(
            # 输入3通道，卷积核3*3，输出64通道（如32*32*3的样本图片，(32+2*1-3)/1+1=32，输出32*32*64）
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 输入64通道，卷积核3*3，输出64通道（输入32*32*64，卷积3*3*64*64，输出32*32*64）
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 输入32*32*64，输出16*16*64
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第二层，2个卷积层和一个最大池化层
        self.layer2 = nn.Sequential(
            # 输入64通道，卷积核3*3，输出128通道（输入16*16*64，卷积3*3*64*128，输出16*16*128）
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 输入128通道，卷积核3*3，输出128通道（输入16*16*128，卷积3*3*128*128，输出16*16*128）
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 输入16*16*128，输出8*8*128
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第三层，3个卷积层和一个最大池化层
        self.layer3 = nn.Sequential(
            # 输入128通道，卷积核3*3，输出256通道（输入8*8*128，卷积3*3*128*256，输出8*8*256）
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 输入256通道，卷积核3*3，输出256通道（输入8*8*256，卷积3*3*256*256，输出8*8*256）
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 输入256通道，卷积核3*3，输出256通道（输入8*8*256，卷积3*3*256*256，输出8*8*256）
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 输入8*8*256，输出4*4*256
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第四层，3个卷积层和1个最大池化层
        self.layer4 = nn.Sequential(
            # 输入256通道，卷积3*3，输出512通道（输入4*4*256，卷积3*3*256*512，输出4*4*512）
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输入512通道，卷积3*3，输出512通道（输入4*4*512，卷积3*3*512*512，输出4*4*512）
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输入512通道，卷积3*3，输出512通道（输入4*4*512，卷积3*3*512*512，输出4*4*512）
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输入4*4*512，输出2*2*512
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第五层，3个卷积层和1个最大池化层
        self.layer5 = nn.Sequential(
            # 输入512通道，卷积3*3，输出512通道（输入2*2*512，卷积3*3*512*512，输出2*2*512）
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输入512通道，卷积3*3，输出512通道（输入2*2*512，卷积3*3*512*512，输出2*2*512）
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输入512通道，卷积3*3，输出512通道（输入2*2*512，卷积3*3*512*512，输出2*2*512）
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输入2*2*512，输出1*1*512
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 1024)
        )

    def forward(self, x, y):
        x = self.conv_layer(x)
        x = x.view(x.size(0), 7*7*512)
        d_x = self.fc(x)

        fog_level = self.task1(d_x)
        vis_ditance = self.task2(d_x, fog_level)
        return fog_level, vis_ditance
