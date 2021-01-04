import torch
import torch.nn as nn

class Relative_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y, y_pre):
        return torch.mean(torch.pow((y - y_pre), 2)) / y