import torch
import torch.nn as nn
from my_model.classifierNet import claasifierNet1
import torch.nn.functional as F

'''rgbd_Net
four channel network
intput : 448*448*4
output : 1024
keypoint:
(1)hole convolution
(2)Global Average pooling
(3)channel fusion
'''


class rgbdNet2(nn.Module):
    def __init__(self, in_chanel = 3, out_chanel = 1024):
        super(rgbdNet2, self).__init__()
        self.in_chanel = in_chanel
        self.out_chanel = out_chanel
        self.FeatureMap = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=4, padding=4,dilation=2),
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
        self.linear = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features=7*7*384, out_features=out_chanel),
            nn.ELU(inplace=True)
        )
    def forward(self, x):
        x = self.FeatureMap(x)
        x= x.view(x.size(0), 7*7*384)
        x = self.linear(x)
        return x


'''
使用GAP的RGBD单特征网络结构rgbdNet
'''


class testrgbdNet2(nn.Module):
    def __init__(self):
        super(testrgbdNet2, self).__init__()
        self.rgbd = rgbdNet2()
        self.classifier = claasifierNet1(input_channel=1)

    def forward(self, x):
        x = self.rgbd(x)
        x = torch.reshape(x, (x.size(0), 1, 32, 32))
        x = self.classifier(x)
        return x
