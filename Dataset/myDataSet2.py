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
    def __init__(self, root, file_rgb, file_d,file_slice, datatxt = '', tranform = None, target_transform = None):
        super(MyDataSet, self).__init__()
        fh = open(os.path.join(datatxt), 'r')
        labels = []
        for line in fh.readlines():
            tlist = line.strip().split(' ')
            if len(tlist) > 3:
                t = ''
                for pp in range(len(tlist) -3):
                    t += tlist[pp]
                    t += ' '
                t += tlist[-3]
                labels.append((t, int(tlist[-2]), float(tlist[-1])))
            else:
                labels.append((tlist[0], int(tlist[1]), float(tlist[2])))
        self.root = root
        self.labels = labels
        self.transform  = tranform
        self.target_transform  = target_transform
        self.file_rgb = file_rgb
        self.file_d = file_d
        self.file_slice = file_slice

    def __getitem__(self, index):
        img_name, label1, label2 = self.labels[index]
        img = Image.open(os.path.join(self.root, self.file_rgb, img_name)).convert('RGB')
        img = self.transform(img)

        img2_ = Image.open(os.path.join(self.root, self.file_d, img_name)).convert('L')
        transf = transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])
        img2_ = transf(img2_)
        img = torch.cat((img, img2_), dim=0).float()

        trans = transforms.Compose(
            [transforms.RandomRotation(20), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img_ = Image.open(os.path.join(self.root, self.file_slice, str(0)+'_'+img_name)).convert('RGB')
        img_ = trans(img_)
        res = img_

        return img_name, img, res, label1, label2


    def __len__(self):
        return len(self.labels)

