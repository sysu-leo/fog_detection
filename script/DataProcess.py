import cv2
import numpy as np
import random
import torch
from PIL import Image
from torchvision import transforms



'''
《Bag of Tricks for Image Classification with Convolution Neural Network》
data enhance trick：
1. Ramdomly crop
2. Flip horizontal
3. Scale hue, saturation, and brightness with coefficients uniformly drawn from [0.6, 1.4]
4. Add PCA noise with a coefficient sampled from a normal distribution N(0, 0.1)
5. Normalize RGB channels by subtracting 123.68, 116.779, 103.939 and dividing by 58.393, 57.12, 57.375Xavirr
6. Learning Rate warmup
7. label smoothing
8. Cosine Learning rate Decay
9. NAG Descent

'''
def create_dark(src, r = 5):
    v = np.min(src, 2)
    cv2.erode(v, np.ones((2 * r + 1, 2 * r + 1)))
    return torch.tensor(v)

def getCanny(src):
    return torch.tensor(cv2.Canny(np.array(src), threshold1=30, threshold2=100))

def slcing(src):

    src_canny = getCanny(src)
    tlist = []
    i = 0
    while i < 10:
        random.seed()
        is_in = False
        random_num_x = random.randrange(0, src.shape[1]-112)
        random_num_y = random.randrange(0, src.shape[0]-112)
        if len(tlist) == 0:
            tlist.append((random_num_x, random_num_y))
            i += 1
        else:
            for t in tlist:
                if (random_num_x> t[0] and random_num_x < t[0] + 112) or ((random_num_y> t[1] and random_num_y < t[1] + 112)):
                    is_in = True
                    break
            if is_in == False:
                tlist.append((random_num_x, random_num_y))
                i += 1
    t_weight = {}

    for t in tlist:
        weight = 0
        for i in range(t[0], t[0]+112):
            for j in range(t[1], t[1] + 112):
                if src_canny[j][i] != 0:
                    weight += 1
        while weight in t_weight.keys():
            weight -= 1
        t_weight[weight] = t


    temp_list = sorted(t_weight.keys())
    res_list = []
    for k in range(0, 3):
        t_ = t_weight[temp_list[k]]
        res_list.append(src[t_[1]:t_[1] + 112,t_[0]: t_[0]+ 112, :])

    for k in range(7, 10):
        t_ = t_weight[temp_list[k]]
        res_list.append(src[t_[1]:t_[1] + 112,t_[0]: t_[0]+ 112, :])
    res = torch.from_numpy(np.array(res_list).transpose((0, 3, 2, 1))*(1/255.0))
    rr = torch.FloatTensor(3, 112, 112)
    for i in range(6):
        if i == 0:
            rr = res[i, :, :, :]
        else:
            rr = torch.cat((rr, res[i, :, :, :]), dim = 0)
    return rr

# PIL与Tensoe互转
def PIL2Tensor(pil_image):
    transform1 = transforms.Compose([transforms.ToTensor()])
    return transform1(pil_image)
def Tensor2PIL(t_tensor):
    return transforms.ToPILImage(t_tensor)

# Tensor 与 cv2 互转
def cv2tensor(src):
    transform1 = transforms.Compose([transforms.ToTensor()])
    return transform1(src)

def tensor2numpy(src):
    src_t = src.mul(255).byte()
    return src.cpu().numpy().squeeze(0).transpose((1, 2, 0))

def minFilter(src, r = 7):
    return cv2.erode(src, np.ones((2*r + 1, 2*r+1)))


