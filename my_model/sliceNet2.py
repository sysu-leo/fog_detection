import torch
import torch.nn as nn
from my_model.classifierNet import claasifierNet1
import torch.nn.functional as F


class sliceNet2(nn.Module):
    def __init__(self, input_channel = 3, out_channel = 1024):
        super(sliceNet2, self).__init__()
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



class testsliceNet2(nn.Module):
    def __init__(self):
        super(testsliceNet2, self).__init__()
        self.slice = sliceNet2(input_channel=4)
        self.classifier = claasifierNet1(input_channel=1)

    def forward(self, x):
        x = self.slice(x)
        x = torch.reshape(x, (x.size(0), 1, 32, 32))
        x = self.classifier(x)
        return x