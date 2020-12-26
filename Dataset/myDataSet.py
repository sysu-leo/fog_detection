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
        fh = open(os.path.join(root, datatxt), 'r')
        labels = []
        for line in fh.readlines():
            tlist = line.strip().split('.')[0].split('-')
            cls = int(tlist[0])
            pre = float(tlist[-1])/ 500.0
            rgb_path = os.path.join(root, 'SCENE_1',str(cls), line.strip())
            dark_path = os.path.join(root, 'SCENE_1_dark',str(cls),  line.strip())
            slice_path = os.path.join(root, 'SCENE_1_slice',str(cls),  line.strip())
            labels.append((line.strip(),rgb_path, dark_path, slice_path, cls, pre))
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

