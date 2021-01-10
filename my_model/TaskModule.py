import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0")
class FogLevel_Classify(nn.Module):

    def __init__(self, input_channel=2):
        super(FogLevel_Classify, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size= 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=7, kernel_size=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    def forward(self, x):
        x = self.classifier(x)
        x = self.classifier2(x)
        x = x.view(x.size(0), 1*1*7)
        return x



class VisDistance_Estimation(nn.Module):

    def __init__(self, input_channel = 2):
        super(VisDistance_Estimation, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size= 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=5, kernel_size=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    def forward(self, x, y):
        x = self.classifier(x)
        x = self.classifier2(x)
        tmp_x = torch.tensor([35.0 / 500.0, 75.0 / 500.0, 150.0 / 500.0, 350.0 / 500.0, 1.0]).reshape((5, 1)).to(device)
        x = x.view(x.size(0), 1 * 1 * 5)
        x = torch.mul(x, y)
        x = torch.mm(x, tmp_x)
        x = torch.squeeze(x)
        return x

class VisDistance_Estimation_2(nn.Module):

    def __init__(self, input_channel = 2):
        super(VisDistance_Estimation_2, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size= 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=5, kernel_size=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.linear_3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=5, out_features=1)
        )

    def forward(self, x):
        x = self.classifier(x)
        x = self.classifier2(x)
        x = x.view(x.size(0), 1 * 1 * 5)
        # print(x.size())
        x = self.linear_3(x)
        x = torch.squeeze(x)
        return x

class FogLevel_Classify_FC(nn.Module):

    def __init__(self, input_channel=1208):
        super(FogLevel_Classify_FC, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.linear_1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=1280, out_features=1024),
            nn.ELU(inplace=True)
        )
        self.linear_2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=256),
            nn.ELU(inplace=True)
        )
        self.linear_3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features = 256, out_features=5)
        )
    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        x = x.view(x.size(0), 1*1*5)
        return x

class Visibility_Estimation_FC(nn.Module):

    def __init__(self, input_channel=1208):
        super(Visibility_Estimation_FC, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.linear_1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=1280, out_features=1024),
            nn.ELU(inplace=True)
        )
        self.linear_2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=256),
            nn.ELU(inplace=True)
        )
        self.linear_3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features = 256, out_features=5)
        )
    def forward(self, x, y):
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        tmp_x = torch.tensor([35.0 / 500.0, 75.0 / 500.0, 150.0 / 500.0, 350.0 / 500.0, 1.0]).reshape((5, 1)).to(device)
        x = x.view(x.size(0), 1 * 1 * 5)
        y = self.softmax(y)
        x = torch.mul(x, y)
        x = torch.mm(x, tmp_x)
        x = torch.squeeze(x)
        return x




