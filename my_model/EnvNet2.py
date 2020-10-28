import torch
import torch.nn as nn
from my_model.classifierNet import claasifierNet1
from my_model.sliceNet2 import sliceNet2
from my_model.rgbdNet2 import rgbdNet2


'''
每个通道生成一个32*32的featuremap; 然后利用基于GAP的分类网络进行分类。
'''
class EnvNet2(nn.Module):
    def __init__(self):
        super(EnvNet2, self).__init__()
        self.rgbdNet = rgbdNet2()
        self.sliceNet = sliceNet2(input_channel=4)
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(2),
            claasifierNet1(input_channel=2)
        )

    def forward(self, x2, x1):

        x1 = self.rgbdNet(x1)
        x2 = self.sliceNet(x2)
        x1 = torch.reshape(x1, (x1.size(0), 1, 32, 32))
        x2 = torch.reshape(x2, (x2.size(0), 1, 32, 32))

        x = torch.cat((x2, x1), dim=1)
        x = self.classifier(x)
        return x
