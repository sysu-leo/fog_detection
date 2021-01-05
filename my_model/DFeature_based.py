import torch
import torch.nn as nn
from my_model.FeatureModule import g_exactor, d_exactor
from my_model.TaskModule import FogLevel_Classify, VisDistance_Estimation

class Multi_Task(nn.Module):
    def __init__(self):
        super(Multi_Task, self).__init__()
        self.d_feature = g_exactor(in_chanel=3)
        self.task1 = FogLevel_Classify(input_channel=1)
        self.task2 = VisDistance_Estimation(input_channel=1)

    def forward(self, x1, x2):
        d_x = self.d_feature(x2)
        d_x = torch.reshape(d_x, (d_x.size(0), 1, 32, 32))

        fog_level = self.task1(d_x)
        vis_ditance = self.task2(d_x)
        return fog_level, vis_ditance