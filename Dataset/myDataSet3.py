from PIL import Image
import os
import torch.utils.data as data
import cv2
import numpy as np
import torch
import  script.DataProcess as dp
from torchvision import transforms

def minFilter(src, r = 7):
    return cv2.erode(src, np.ones((2*r + 1, 2*r+1)))


class MyDataSet(data.Dataset):
    def __init__(self, root, datatxt, tranform = None, target_transform = None):
        super(MyDataSet, self).__init__()
        fh = open(datatxt, 'r')
        labels = []
        for line in fh.readlines():
            tlist = line.strip().split('.')[0].split('_')
            cls = -1
            if int(tlist[-1]) == 50:
                cls = 0
            elif int(tlist[-1]) == 100:
                cls = 1
            elif int(tlist[-1]) == 150:
                cls = 2
            elif int(tlist[-1]) == 200:
                cls = 3
            elif int(tlist[-1]) == 250:
                cls = 4
            elif int(tlist[-1]) == 300:
                cls = 5
            elif int(tlist[-1]) == 400:
                cls = 6


            pre = float(int(tlist[-1]))/ 500.0
            rgb_path = os.path.join(root, 'FROSI', line.strip())
            dark_path = os.path.join(root, 'DARK_FROSI',  line.strip())
            slice_path = os.path.join(root, 'SLICE_FROSI',str(cls),  line.strip())
            labels.append((line.strip(),rgb_path, dark_path, cls, pre))
        self.root = root
        self.labels = labels
        self.transform  = tranform
        self.target_transform  = target_transform

    def __getitem__(self, index):
        img_name, rgb_path, dark_path, slice_path, label1, label2 = self.labels[index]
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_img = self.transform(rgb_img)

        dark_img = Image.open(dark_path).convert('L')
        transf = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])
        dark_img = transf(dark_img)
        img = torch.cat((rgb_img, dark_img), dim=0).float()

        trans = transforms.Compose(
            [transforms.RandomRotation(20), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        slice_img = Image.open(slice_path).convert('RGB')
        slice_img = trans(slice_img)
        img2 = slice_img

        return img_name, img, img2, label1, label2


    def __len__(self):
        return len(self.labels)

