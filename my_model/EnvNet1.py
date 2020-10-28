import torch
import torch.nn as nn
from my_model.classifierNet import claasifierNet1
from my_model.sliceNet1 import sliceNet
from my_model.rgbdNet1 import rgbdNet1


'''
每个通道生成一个32*32的featuremap; 然后利用基于GAP的分类网络进行分类。
'''
class EnvNet1(nn.Module):
    def __init__(self):
        super(EnvNet1, self).__init__()
        self.rgbdNet = rgbdNet1()
        self.sliceNet = sliceNet()
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(2),
            claasifierNet1(input_channel=2)
        )

    def forward(self, x1, x2):

        x1 = self.rgbdNet(x1)
        x2 = self.sliceNet(x2)
        x1 = torch.reshape(x1, (x1.size(0), 1, 32, 32))
        x2 = torch.reshape(x2, (x2.size(0), 1, 32, 32))

        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x
