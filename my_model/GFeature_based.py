import torch
import torch.nn as nn
from my_model.FeatureModule import g_exactor, d_exactor
from my_model.TaskModule import FogLevel_Classify_FC, Visibility_Estimation_FC

class Multi_Task(nn.Module):
    def __init__(self):
        super(Multi_Task, self).__init__()
        self.g_feature = g_exactor()
        self.task1 = FogLevel_Classify_FC(input_channel=1024)
        self.task2 = Visibility_Estimation_FC(input_channel=1024)

    def forward(self, x1, x2):
        g_x = self.g_feature(x1)

        fog_level = self.task1(g_x)
        vis_ditance = self.task2(g_x, fog_level)
        return fog_level, vis_ditance