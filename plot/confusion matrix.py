import torch
import torchvision
import numpy as np
import seaborn as sns
import h5py
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from matplotlib import rcParams
# 导入网络（根据实际需要选择）
from resnet1d import resnet18_1d
from resnet import resnet18
from vggnet2d import vgg2d
from vggnet1d import vgg1d
from alexnet import AlexNet
from alexnet1d import AlexNet1d
from googlenet import GoogLeNet
from googlenet1d import GoogLeNet1d

# 初始化网络
#net = vgg1d()
net = vgg2d()
#net = resnet18_1d()
#net = resnet18()
#net = AlexNet()
#net = AlexNet1d()
#net = GoogLeNet()
#net = GoogLeNet1d()

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)  # 将网络移动到GPU


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


# 类别名称和参数设置
#class_names = ['ball', 'inner', 'outer', 'health','ball2', 'inner2', 'outer2']
#class_names = ['ball', 'inner', 'outer', 'ball2', 'inner2', 'outer2']
class_names = ['B007','B014','B021', 'IR007', 'IR014', 'IR021', 'normal', 'OR007', 'OR014', 'OR021']
BATCH_SIZE = 200

# 加载测试数据
testloader = DataLoader(
    H5Dataset('C://Users//zh//Desktop//train//gasf_val_dataset.h5'),
    BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# 加载模型权重（自动处理设备映射）
net.load_state_dict(torch.load(
    'C://Users//zh//Desktop//train//vggnet/gasf//net.pth',
    map_location=device  # 添加设备映射
))


# 预测函数（GPU版本）
def get_predictions(net, testloader):
    # 设置学术图表全局参数
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['axes.linewidth'] = 1.5
    net.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for images, labels in testloader:
            # 将数据移动到GPU
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = net(images)

            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)

            # 将结果移回CPU进行处理
            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return predictions, targets


# 生成预测结果
predictions, targets = get_predictions(net, testloader)

# 生成混淆矩阵
confusion_mat = confusion_matrix(targets, predictions)


# 可视化函数（保持不变）
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrBr",  # 改用黄色系色盘
        square=True,
        annot_kws={"size": 30, "color": "black", "weight": "bold"},  # 加粗字体
        cbar_kws={"shrink": 0.8,
                  "drawedges": False
                  }  # 添加颜色条设置
    )
    # 动态调整注释颜色
    for i, text in enumerate(ax.texts):
        row = i // cm.shape[1]
        col = i % cm.shape[1]

        # 对角线元素使用深色背景+白色文字
        if row == col:
            text.set_color("white")
            text.set_path_effects([
                PathEffects.withStroke(linewidth=2, foreground="black")  # 添加文字描边
            ])
        # 非对角线元素使用黑色文字
        else:
            text.set_color("black")
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)  # 外边框线宽
        spine.set_color("black")  # 与标签颜色一致


    # 设置标题
    ax.set_title("", fontsize=18, pad=20)

    # 坐标轴标签样式
    ax.set_xlabel("Predicted Label", fontsize=24, labelpad=15,)
    ax.set_ylabel("True Label", fontsize=24, labelpad=15)

    # 坐标轴刻度样式
    ax.xaxis.set_ticklabels(classes, rotation=45, ha="right", fontsize=20,fontstyle='normal')
    ax.yaxis.set_ticklabels(classes, rotation=0, fontsize=20,fontstyle='normal')

    # 设置刻度线位置和颜色
    ax.tick_params(axis='both', which='both', length=0, colors="dimgrey")
    # 颜色条调整
    cbar = ax.collections[0].colorbar
    cbar.outline.set_linewidth(1.5)
    cbar.ax.tick_params(
        length=0,
        labelsize=20,
    )
    plt.tight_layout()
    plt.show()


# 显示混淆矩阵
plot_confusion_matrix(confusion_mat, class_names)