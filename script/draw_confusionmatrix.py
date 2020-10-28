from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库
import numpy as np

def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_line(x_axix, train_acc, valid_acc, train_loss, valid_loss):
    plt.title('Result Analysis')
    plt.plot(x_axix, train_acc, color='green', label='training accuracy')
    plt.plot(x_axix, valid_acc, color='red', label='testing accuracy')
    plt.plot(x_axix, train_loss, color='skyblue', label='PN distance')
    plt.plot(x_axix, valid_loss, color='blue', label='threshold')
    plt.legend()  # 显示图例

    plt.xlabel('iteration times')
    plt.ylabel('rate')
    plt.show()

def readTrain_data(path):
    tlist  = open(path, 'r').readlines()
    loss = []
    acc = []
    count = 0
    for i in tlist:
        count +=1
        tt = i.strip().split(' ')
        loss.append(float(tt[1]))
        acc.append(float(tt[2]))
        if count > 40:
            break
    return loss, acc
def loss_acc_compare():
    path1 = '/mnt/26C00662C0063897/fog image classification/train_data/valid_FC-Net.txt'
    path2 = '/mnt/26C00662C0063897/fog image classification/train_data/valid_efficient.txt'
    path3 = '/mnt/26C00662C0063897/fog image classification/train_data/valid_efficient1.txt'
    path4 = '/mnt/26C00662C0063897/fog image classification/train_data/valid_efficient2.txt'
    path5 = '/mnt/26C00662C0063897/fog image classification/train_data/valid_efficient7.txt'
    path6 = '/mnt/26C00662C0063897/fog image classification/train_data/valid_inception.txt'
    path7 = '/mnt/26C00662C0063897/fog image classification/train_data/valid_inception_res.txt'
    loss1, acc1 = readTrain_data(path1)
    loss2, acc2 = readTrain_data(path2)
    loss3, acc3 = readTrain_data(path3)
    loss4, acc4 = readTrain_data(path4)
    loss5, acc5 = readTrain_data(path5)
    loss6, acc6 = readTrain_data(path6)
    loss7, acc7 = readTrain_data(path7)
    x_axis = [i for i in range(len(loss1))]
    plt.plot(x_axis, acc1, color='green', label='valid_FC-Net1 acc')
    plt.plot(x_axis, acc2, color='greenyellow', label='valid_efficient acc')
    plt.plot(x_axis, acc3, color='palegreen', label='valid_efficient1 acc')
    plt.plot(x_axis, acc4, color='darkred', label='valid_efficient2 acc')
    plt.plot(x_axis, acc5, color='crimson', label='valid_efficient7 acc')
    plt.plot(x_axis, acc6, color='red', label='valid_inception acc')
    plt.plot(x_axis, acc7, color='orchid', label='valid_inception_res acc')
    plt.legend()
    plt.xlabel('iteration times')
    plt.ylabel('rate')
    plt.show()
def loss_acc_compare2():
    path1 = '/mnt/26C00662C0063897/fog image classification/train_data/train_FC-Net1.txt'
    path2 = '/mnt/26C00662C0063897/fog image classification/train_data/train_slice1.txt'
    path3 = '/mnt/26C00662C0063897/fog image classification/train_data/train_rgbd1.txt'
    loss1, acc1 = readTrain_data(path1)
    loss2, acc2 = readTrain_data(path2)
    loss3, acc3 = readTrain_data(path3)
    x_axis = [i for i in range(len(loss1))]
    plt.plot(x_axis, acc1, color='green', label='train_FC-Net1 acc')
    plt.plot(x_axis, acc2, color='greenyellow', label='train_slice acc')
    plt.plot(x_axis, acc3, color='palegreen', label='train_rgbd1 acc')
    plt.legend()
    plt.xlabel('iteration times')
    plt.ylabel('rate')
    plt.show()
loss_acc_compare2()



