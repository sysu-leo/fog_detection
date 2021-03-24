import torch
import torch.nn as nn
from my_model.FeatureModule import g_exactor, d_exactor
from my_model.TaskModule import FogLevel_Classify, VisDistance_Estimation, FogLevel_Classify_FC, Visibility_Estimation_FC_na, VisDistance_Estimation_2

class DisEstimation_Single(nn.Module):
    def __init__(self):
        super(DisEstimation_Single, self).__init__()
        self.g_feature = g_exactor()
        self.d_feature = d_exactor(input_channel=3, out_channel=256)
        self.task1 = FogLevel_Classify_FC()
        self.task2 = Visibility_Estimation_FC_na()

    def forward(self, x1, x2):

        g_x = self.g_feature(x1)
        d_x = self.d_feature(x2)

        m_x = torch.cat((d_x, g_x), dim=1)

        fog_level = self.task1(m_x)
        vis_ditance = self.task2(m_x, fog_level)

        return vis_ditance