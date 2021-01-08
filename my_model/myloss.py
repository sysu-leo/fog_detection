import torch
import torch.nn as nn

class Relative_loss(nn.Module):
    def __init__(self):
        super(Relative_loss, self).__init__()
        self.l1 = nn.MSELoss()
        self.l2 = nn.CrossEntropyLoss()
        
    def forward(self, y, y_pre, x, x_cls):
        sum  = 1*self.l1(y, y_pre) + 1*self.l2(x, x_cls)
        return sum