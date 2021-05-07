import matplotlib.pyplot as plt
import numpy as np


def plot_Matrix(cm, classes, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='Times New Roman', size='12', weight = 'bold')  # 设置字体样式、大小

    # 按行进行归一化
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         if int(cm[i, j] * 100) == 0:
    #             cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title)

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = '.1f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if (cm[i, j] * 100) > 0.5:
                ax.text(j, i, format((cm[i, j] * 100), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('cm.jpg', dpi=1000)
    plt.rcParams['figure.dpi'] = 1000
    plt.show()

import os
path = '/home/liqiang/Desktop/GB'
cm = np.zeros(shape=(5, 5))

file = open(path, 'r')
a = 0
for i in file.readlines():
    b = 0
    for t in i.strip().split('\t'):
        cm[a][b] = float(t)
        b += 1
    a += 1

cls = ['clear', 'low', 'mid', 'high', 'dense']
plot_Matrix(cm, cls, title='VENet-NK')


