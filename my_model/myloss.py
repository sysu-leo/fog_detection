import torch
import torch.nn as nn

class Relative_loss(nn.Module):
    def __init__(self):
        super(Relative_loss, self).__init__()
        
    def forward(self, y, y_pre):
        pp = torch.div(torch.abs(y - y_pre), y)
        sum = torch.mean(pp)
        return sum