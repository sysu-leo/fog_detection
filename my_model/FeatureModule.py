import torch
import torch.nn as nn
import torch.nn.functional as F

'''rgbd_feature_exactor
four channel network
intput : 448*448*4
output : 1024
keypoint:
(1)hole convolution
(2)Global Average pooling
(3)channel fusion
'''


class g_exactor(nn.Module):
    def __init__(self, in_chanel = 4, out_chanel = 1024):
        super(g_exactor, self).__init__()
        self.in_chanel = in_chanel
        self.out_chanel = out_chanel
        self.FeatureMap = nn.Sequential(
            nn.Conv2d(in_channels=in_chanel, out_channels=96, kernel_size=5, stride=4, padding=4,dilation=2),
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

class d_exactor(nn.Module):
    def __init__(self, input_channel = 3, out_channel = 1024):
        super(d_exactor, self).__init__()
        self.input_channel = input_channel
        self.out_channel  = out_channel
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
            nn.Dropout(p=0.4),
            nn.Linear(in_features=7*7*384, out_features=self.out_channel),
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

