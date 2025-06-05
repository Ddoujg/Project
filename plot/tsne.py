import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import h5py

def h5():
    """基础读取方法（适合小规模数据）"""
    with h5py.File(h5_path, 'r') as hf:
        data = hf['data'][:]  # 获取全部数据
        labels = hf['labels'][:]  # 获取全部标签
        return data, labels
# ====================================================`
# 0. 基础配置（根据实际情况修改路径和参数）
# ====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
h5_path = 'origin_val_dataset.h5'
batch_size = 200                      # 特征提取的批大小
# ====================================================
# 1. 直接导入训练时的模型定义（关键步骤！）
# ====================================================
from resnet1d import resnet18_1d  # 替换为实际模块和类名
from resnet import resnet18
from vggnet2d import vgg2d
from vggnet1d import vgg1d
from alexnet import AlexNet
from alexnet1d import AlexNet1d
from googlenet import GoogLeNet
from googlenet1d import GoogLeNet1d

model = resnet18_1d()  # 初始化模型
#model = resnet18()
#model = vgg1d().to(device)
#model = vgg2d().to(device)
#model = AlexNet().to(device)
#model = AlexNet1d().to(device)
#model = GoogLeNet().to(device)
#model = GoogLeNet1d().to(device)
model.load_state_dict(torch.load('net.pth'))  # 加载权重


# ====================================================
# 2. 修改模型为特征提取模式
# ====================================================
# 移除分类层，假设原模型最后一层是 fc
if hasattr(model, 'fc'):
    model.fc = torch.nn.Identity()  # 直接输出倒数第二层的特征

# 设置为评估模式
model.eval().to(device)

# ====================================================
# 3. 加载数据和标签
# ====================================================
# 加载数据（假设数据是张量或NumPy数组）
data,labels = h5()
# 统一转换为 PyTorch 张量并送至设备
if isinstance(data, np.ndarray):
    data = torch.from_numpy(data).float()  # NumPy数组转张量
data = data.to(device)

# 标签处理（转换为numpy数组）
if isinstance(labels, torch.Tensor):
    labels = labels.cpu().numpy()
elif isinstance(labels, np.ndarray):
    labels = labels.astype(int)  # 确保标签为整数类型

# 检查数据形状
print(f"2D数据形状: {data.shape}")  # 应类似 (n_samples, channels, height, width)
print(f"标签形状: {labels.shape}")     # 应类似 (n_samples,

# ====================================================
# 4. 特征提取
# ====================================================
def extract_features(model, data, batch_size):
    features = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            features.append(model(batch).cpu().numpy())
    return np.concatenate(features, axis=0)

features = extract_features(model, data, batch_size)

# ====================================================
# 5. 特征标准化与保存
# ====================================================
features_scaled = StandardScaler().fit_transform(features)
# np.save("2d_features.npy", features_scaled)
# np.save("2d_labels.npy", labels)

# ====================================================
# 6. t-SNE可视化
# ====================================================
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
tsne_results = tsne.fit_transform(features_scaled)


# 全局设置学术图表格式（字体、字号、刻度等）
plt.rcParams.update({
    "font.family": "serif",        # 使用衬线字体（如Times New Roman）
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 20,          # 坐标轴标签字号
    "xtick.labelsize": 20,         # x轴刻度标签字号
    "ytick.labelsize": 20,         # y轴刻度标签字号
    "xtick.direction": "in",       # 刻度线向内
    "ytick.direction": "in",       # 刻度线向内
    "axes.linewidth": 1.0,         # 坐标轴线宽
    "legend.fontsize": 20,         # 图例字号（如果有）
})

plt.figure(figsize=(8, 6))  # 根据期刊要求调整尺寸（单位：英寸）

# 绘制散点图
scatter = plt.scatter(
    tsne_results[:, 0],
    tsne_results[:, 1],
    c=labels,
    cmap="tab10",
    s=50,
    alpha=0.8,
    edgecolor="face",  # 添加黑色边缘提高区分度
    linewidth=0.5
)

# 坐标轴标签（包含单位或说明）
plt.xlabel("t-SNE dimension1", labelpad=10)  # labelpad调整标签与坐标轴的间距
plt.ylabel("t-SNE dimension2", labelpad=10)

# 调整刻度格式（学术图通常不需要四周的边框）
plt.tick_params(axis='both', which='both', top=False, right=False)

# 颜色条格式
cbar = plt.colorbar(scatter, pad=0.03)  # pad控制颜色条与主图的间距
cbar.set_label("Class", fontsize=20, labelpad=10)
cbar.ax.tick_params(labelsize=20, direction="in")  # 刻度线向内

plt.show()
