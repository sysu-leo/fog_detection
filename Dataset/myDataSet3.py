
from PIL import Image
import os
import torch.utils.data as data
import cv2
import numpy as np
import torch
from torchvision import transforms

def minFilter(src, r = 7):
    return cv2.erode(src, np.ones((2*r + 1, 2*r+1)))


class MyDataSet(data.Dataset):
    def __init__(self, root, file_rgbd, file_dark,file_slice,  datatxt , tranform = None, target_transform = None, is_slice = False):
        super(MyDataSet, self).__init__()
        fh = open(os.path.join(datatxt), 'r')
        labels = []
        for line in fh.readlines():
            tlist = line.strip().split(' ')
            if len(tlist) > 2:
                t = ''
                for pp in range(len(tlist) -2):
                    t += tlist[pp]
                    t += ' '
                t += tlist[-2]
                labels.append((t, int(tlist[-1])))
            else:
                labels.append((tlist[0], int(tlist[1])))
        self.root = root
        self.labels = labels
        self.transform  = tranform
        self.target_transform  = target_transform
        self.file = file_rgbd
        self.file2 = file_dark
        self.file3 = file_slice
        self.is_slice = is_slice

    def __getitem__(self, index):
        img_name, label = self.labels[index]
        img = Image.open(os.path.join(self.root, self.file, img_name)).convert('RGB')
        img = self.transform(img)
        return img_name, img, label

    def __len__(self):
        return len(self.labels)






