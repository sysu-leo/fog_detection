import torch
from torch.autograd import Variable
from torch.autograd import  Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse

class FeatureExtractor():

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layer = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self,x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layer:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class Model