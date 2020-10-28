from torch import nn
from torch.nn import functional as F
import os
import sys

current_dir = os.getcwd()    # obtain work dir
sys.path.append(current_dir) # add work dir to sys path

class LeNet(nn.Module):
    def __init__(self, num_class):
        super(LeNet, self).__init__()
        self.num_classes= num_class
        self.conv1 = nn.Conv2d(3, 96, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(96, 256, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
