import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# 保证图片在浏览器内正常显示
file = open("/home/liqiang/Downloads/VIO/CurveFitting_LM/build/app/res.txt", 'r')
x = []
y = []
for i in file.readlines():
    tlist = i.split(' ')
    x.append(int(tlist[0]))
    y.append(float(tlist[1]))
plt.plot(x, y)
plt.xlabel('iteration')
plt.ylabel('lambda')
plt.show()