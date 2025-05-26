"""
Created on Wed Aug  5 14:09:35 2020

@author: 17505
一维识别
"""
import time
import os
from pydoc import importfile

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import scipy.io as sio
from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import h5py
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
from torchvision.models import vgg16
from torchvision.models.quantization import googlenet

#导入网络
from resnet import resnet18
from resnet1d import resnet18_1d
from vggnet2d import vgg2d
from vggnet1d import vgg1d
from alexnet import AlexNet
from alexnet1d import AlexNet1d
from googlenet import GoogLeNet
from googlenet1d import GoogLeNet1d
class H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5 = h5py.File(h5_path, 'r')
        self.data = self.h5['data']
        self.labels = self.h5['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.data[idx]).float(),
            torch.tensor(self.labels[idx]).long()
        )



###训练###

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型定义
#net = vgg1d().to(device)
#net = vgg2d().to(device)
#net = resnet18_1d().to(device)
net = resnet18().to(device)
#net = AlexNet().to(device)
#net = AlexNet1d().to(device)
#net = GoogLeNet().to(device)
#net = GoogLeNet1d().to(device)
# 超参数设置
start = time.time()
EPOCH = 100  #遍历数据集次数
BATCH_SIZE = 200      #批处理尺寸(batch_size)
LR = 0.000003       #学习率  0.0000008 0.0000005 0.0000003
# 数据包装
trainloader = DataLoader(H5Dataset('fmd_gasf_train_dataset.h5'), batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(H5Dataset('fmd_gasf_val_dataset.h5'),batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

L1 = len(trainloader)
L2 = len(testloader)
C1 = len(trainloader.dataset)
C2 = len(testloader.dataset)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.Adam(net.parameters(), lr=LR)

# 训练
sLoss_list = []
vLoss_list = []
sCorrect_list = []
vCorrect_list = []
best_correct = 0

save_path = './net.pth'

print("Start Training")  # 定义遍历数据集的次数
for epoch in range(EPOCH):
    
    if epoch % 10 == 0:
        LR = LR*0.95
    
    net.train()
    s_loss = 0.0
    v_loss = 0.0
    s_correct = 0.0
    s_total = 0.0
    for i, data in enumerate(trainloader, 0):
        length = len(trainloader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device=device, dtype=torch.int64)
        optimizer.zero_grad()
        ####### googlenet训练
        # # 前向传播
        # outputs, aux2, aux1 = net(inputs)  # 获取三个输出
        #
        # # 计算各损失
        # loss_main = criterion(outputs, labels)
        # loss_aux1 = criterion(aux1, labels)
        # loss_aux2 = criterion(aux2, labels)
        #
        # # 加权求和（论文中权重为0.3）
        # loss = loss_main + 0.3 * (loss_aux1 + loss_aux2)
        # # 反向传播
        # loss.backward()
        # optimizer.step()
        ######### 其他方法训练
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 每训练1个batch打印一次loss和准确率
        s_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        s_total += labels.size(0)
        s_correct += predicted.eq(labels.data).cpu().sum()
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
        % (epoch + 1, (i + 1 + epoch * length), s_loss / (i + 1), 100. * s_correct / s_total))
    sCorrect_list.append(100 * s_correct/C1)

        # 每训练完一个epoch测试一下准确率
    print("Waiting Test!")
    with torch.no_grad():
        v_correct = 0
        v_total = 0
        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device=device, dtype=torch.int64)
            outputs = net(images)
            val_loss = criterion(outputs, labels)
            v_loss += val_loss.item()
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            v_total += labels.size(0)
            v_correct += (predicted == labels).sum()
        print('训练分类准确率为：%.3f%%' % (100 * torch.true_divide(s_correct, s_total)))  # v_correct / v_total
        print('测试分类准确率为：%.3f%%' % (100 * torch.true_divide(v_correct, v_total) )) #v_correct / v_total

        vCorrect_list.append(100 * torch.true_divide(v_correct, C2) )
        if v_correct > best_correct:
            best_correct = v_correct
            torch.save(net.state_dict(), save_path)
    
    sLoss_list.append(s_loss/L1)
    vLoss_list.append(v_loss/L2)
        
    if (epoch+1) % 1 == 0:
        print('train loss: {:.10f}'.format(s_loss/L1)) 
        print('val loss: {:.10f}'.format(v_loss/L2))  #length

print('finished training')
end = time.time()
print (end-start)
###Loss绘图###
x = range(1, EPOCH+1)
#y1 = np.array(sLoss_list)
#y2 = np.array(vLoss_list)

#y3 = np.array(sCorrect_list)
#y4 = np.array(vCorrect_list)

y1 = sLoss_list
y2 = vLoss_list

y3 = sCorrect_list
y4 = vCorrect_list
y4=torch.tensor(y4, device='cpu')

plt.subplot(2, 1, 1)
plt.plot(x, y1, 'b.-')
plt.plot(x, y2, 'r.-')
plt.title('Loss and Accuracy vs. Epochs')
plt.xlabel('Epoches')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(x, y3, 'bo-')
plt.plot(x, y4, 'ro-')
plt.xlabel('Epoches')
plt.ylabel('Accuracy')

plt.show()
#plt.savefig("accuracy_loss.jpg")

save_fn = 'y1.mat'
save_array = y1
sio.savemat(save_fn, {'y1': save_array})

save_fn = 'y2.mat'
save_array = y2
sio.savemat(save_fn, {'y2': save_array})

y33 = np.array(y3)
save_fn = 'y33.mat'
save_array = y33
sio.savemat(save_fn, {'y33': save_array})

y44 = np.array(y4)
y44 = y44.astype(np.float64)
save_fn = 'y44.mat'
save_array = y44
sio.savemat(save_fn, {'y44': save_array})
