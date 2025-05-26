import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# 设置全局字体
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"  # 数学公式字体配套设置
# 配置参数
signal_labels = ['B007', 'B014', 'B021', 'IR014', 'IR007','IR021', 'normal', 'OR007', 'OR014', 'OR021']
file_names = ['X118_DE_time', 'X185_DE_time', 'X222_DE_time', 'X169_DE_time', 'X105_DE_time','X209_DE_time', 'X097_DE_time', 'X135_DE_time', 'X201_DE_time', 'X238_DE_time']
data_columns = ['X118_DE_time', 'X185_DE_time', 'X222_DE_time', 'X169_DE_time', 'X105_DE_time','X209_DE_time', 'X097_DE_time', 'X135_DE_time', 'X201_DE_time', 'X238_DE_time']
# signal_labels = ['ball', 'inner', 'outer','ball2', 'inner2', 'outer2']
# file_names =['ball', 'inner', 'outer','ball2', 'inner2', 'outer2']
# data_columns = ['ball', 'iner', 'outer','ball2', 'inner2', 'outer2']
# signal_labels = ['ball', 'inner', 'outer','ball2', 'inner2', 'outer2']
# file_names =['ball.mat', 'inner.mat', 'outer.mat','ball2.mat', 'inner2.mat', 'outer2.mat']
# data_columns = ['ball', 'inner', 'outer','ball2', 'inner2', 'outer2']
Fs = 12000  # 采样频率
N = 24000 # 截取长度

# 存储频率和振幅数据
freq_list = []
amp_list = []

# 数据加载与处理
for idx in range(10):
    # 加载数据文件
    mat_data = loadmat(f'C:\\Users\\zh\\Desktop\\西储原始信号\\{file_names[idx]}')
    raw_signal = mat_data[data_columns[idx]].flatten()[:N]

    # 计算FFT
    fft_result = np.fft.fft(raw_signal)
    n = len(raw_signal)

    # 生成单边频谱
    freq = np.fft.fftfreq(n, d=1 / Fs)[:n // 2]
    amplitude = 2 / n * np.abs(fft_result[:n // 2])

    freq_list.append(freq)
    amp_list.append(amplitude)

# 创建3D图形
fig = plt.figure(figsize=(18, 12))
ax = fig.add_subplot(111, projection='3d')

# 生成Y轴位置（每个信号对应一个平面）
y_positions = np.arange(10)  # 0-4对应5个信号

# 绘制3D曲面
for y, (freq, amp, label) in enumerate(zip(freq_list, amp_list, signal_labels)):
    # 创建网格：X=freq, Y=constant, Z=amp
    X = freq
    Y = np.full_like(X, y)  # Y轴固定为该信号位置
    Z = amp

    # 绘制线框
    ax.plot(X, Y, Z,
            lw=1.5,
            label=label,
            alpha=0.8)

    # 可选：添加平面投影
    ax.plot(X, Y, np.zeros_like(Z),
            color='gray',
            alpha=0.1,
            linestyle='--')

# 图形装饰
ax.set_xlabel('Frequency (Hz)', fontsize=20, labelpad=20)
ax.set_ylabel('', fontsize=20)
ax.set_zlabel('Amplitude', fontsize=20, labelpad=15, rotation=90)
ax.zaxis.set_rotate_label(False)
ax.zaxis._axinfo['juggled'] = (1, 2, 0)
# 设置Y轴刻度标签
ax.set_yticks(y_positions)
ax.set_yticklabels(signal_labels)
ax.tick_params(axis='x',labelsize=20, pad=3)
ax.tick_params(axis='y',labelsize=20, pad=0,rotation=75)
ax.tick_params(axis='z',labelsize=20, pad=5)
# Modified: 设置坐标平面边框颜色
ax.xaxis.pane.set_edgecolor('k')
ax.yaxis.pane.set_edgecolor('k')
ax.zaxis.pane.set_edgecolor('k')
# 设置视角
ax.view_init(elev=22, azim=-35)  # 俯仰角22度，方位角-35度
# 设置网格线（添加细网格辅助观察）
ax.xaxis._axinfo["grid"].update({"linewidth": 0.1, "color": "gray"})  # Modified: 添加细网格
ax.yaxis._axinfo["grid"].update({"linewidth": 0.1, "color": "gray"})
ax.zaxis._axinfo["grid"].update({"linewidth": 0.1, "color": "gray"})
# 设置坐标轴范围
ax.set_xlim(1, Fs / 2)
ax.set_zlim(0, np.max(amp_list) * 1.1)
# 图形美化（优化整体显示）
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.fill = False  # Modified: 保持坐标平面透明
    axis.pane.set_alpha(0.95)  # Modified: 添加轻微透明度
plt.tight_layout()
plt.show()