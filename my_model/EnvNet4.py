import torch
import torch.nn as nn
import torch.nn.functional as F
from my_model.classifierNet import claasifierNet1
from my_model.sliceNet1 import sliceNet
from my_model.rgbdNet1 import rgbdNet1


'''
每个通道生成一个32*32的featuremap; 然后利用基于GAP的分类网络进行分类。
'''

class attentionlayer(nn.Module):
    def __init__(self):
        super(attentionlayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.li1 = nn.Linear(2, 32)
        self.li2 = nn.Linear(32, 32)
        self.li3 = nn.Linear(32, 2)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size(0), 2)
        x = self.li1(x)
        x = self.sig(x)
        x = self.li2(x)
        x = self.sig(x)
        x = self.li3(x)
        x = self.sig(x)
        return x

class EnvNet4(nn.Module):
    def __init__(self):
        super(EnvNet4, self).__init__()
        self.rgbdNet = rgbdNet1()
        self.sliceNet = sliceNet()
        self.att = attentionlayer()

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(2),
            claasifierNet1(input_channel=2)
        )

    def forward(self, x1, x2):

        x1 = self.rgbdNet(x1)
        x2 = self.sliceNet(x2)
        x1 = F.softmax(x1)
        x2 = F.softmax(x2)
        p1 = (torch.reshape(x1, (x1.size(0), 1, 32, 32)))
        p2 = (torch.reshape(x2, (x2.size(0), 1, 32, 32)))
        t = torch.cat(( p1, p2), dim=1)
        at = self.att(t)
        t1 = torch.unsqueeze(at[:, 0], dim=1)
        t2 = torch.unsqueeze(at[:, 1], dim=1)
        x1 = x1 * t1
        x2 = x2 * t2
        x1 = torch.reshape(x1, (x1.size(0), 1, 32, 32))
        x2 = torch.reshape(x2, (x2.size(0), 1, 32, 32))
        x = torch.cat((x1, x2), dim = 1)

        x = self.classifier(x)
        return x
