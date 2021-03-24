import torch
import torch.nn as nn
from my_model.FeatureModule import g_exactor, d_exactor
from my_model.TaskModule import FogLevel_Classify_FC

class FogLevel_Single(nn.Module):
    def __init__(self):
        super(FogLevel_Single, self).__init__()
        self.g_feature = g_exactor()
        self.d_feature = d_exactor(input_channel=3, out_channel=256)
        self.task = FogLevel_Classify_FC()

    def forward(self, x1, x2):
        g_x = self.g_feature(x1)
        d_x = self.d_feature(x2)

        m_x = torch.cat((d_x, g_x), dim=1)

        fog_level = self.task(m_x)
        return fog_level


