import torch
import torch.nn as nn
from my_model.FeatureModule import g_exactor, d_exactor
from my_model.TaskModule import FogLevel_Classify, VisDistance_Estimation, FogLevel_Classify_FC, Visibility_Estimation_FC

class Multi_Task(nn.Module):
    def __init__(self):
        super(Multi_Task, self).__init__()
        self.g_feature = g_exactor(in_chanel=4)
        self.d_feature = g_exactor(in_chanel=3)
        self.task1 = FogLevel_Classify()
        self.task2 = VisDistance_Estimation()

    def forward(self, x1, x2):
        g_x = self.g_feature(x1)
        d_x = self.d_feature(x2)

        g_x = torch.reshape(g_x, (g_x.size(0), 1, 32, 32))
        d_x = torch.reshape(d_x, (d_x.size(0), 1, 32, 32))
        m_x = torch.cat((d_x, g_x), dim=1)

        fog_level = self.task1(m_x)
        vis_ditance = self.task2(m_x, fog_level)

        return fog_level, vis_ditance

class Multi_Task_2(nn.Module) :
    def __init__(self):
        super(Multi_Task_2, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.g_feature = g_exactor(in_chanel=4, out_chanel=1024)
        self.d_feature = g_exactor(in_chanel=3, out_chanel=256)
        self.task1 = FogLevel_Classify_FC()
        self.task2 = Visibility_Estimation_FC()

    def forward(self, x1, x2):
        g_x = self.g_feature(x1)
        d_x = self.d_feature(x2)

        m_x = torch.cat((d_x, g_x), dim=1)

        fog_level = self.task1(m_x)
        vis_ditance = self.task2(m_x, fog_level)

        return fog_level, vis_ditance

class Multi_Task_3(nn.Module) :
    def __init__(self):
        super(Multi_Task_3, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.g_feature = g_exactor(in_chanel=4, out_chanel=1024)
        self.d_feature = g_exactor(in_chanel=3, out_chanel=272)
        self.task1 = FogLevel_Classify(input_channel=1)
        self.task2 = VisDistance_Estimation(input_channel=1)

    def forward(self, x1, x2):
        g_x = self.g_feature(x1)
        d_x = self.d_feature(x2)

        m_x = torch.cat((d_x, g_x), dim=1)
        g_x = torch.reshape(m_x, (m_x.size(0), 1, 36, 36))

        fog_level = self.task1(m_x)
        vis_ditance = self.task2(m_x, fog_level)

        return fog_level, vis_ditance