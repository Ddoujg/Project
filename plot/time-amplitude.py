import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


# 设置全局字体
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"  # 数学公式字体配套设置
# 定义数据列、文件名和信号标签
signal_labels = ['normal','OR014','B007', 'B014', 'B021', 'IR014', 'IR007','OR021','IR021', 'OR007']
file_names = ['X097_DE_time', 'X201_DE_time', 'X118_DE_time', 'X185_DE_time', 'X222_DE_time', 'X169_DE_time', 'X105_DE_time', 'X238_DE_time' ,'X209_DE_time','X135_DE_time']
data_columns = ['X097_DE_time','X201_DE_time','X118_DE_time', 'X185_DE_time', 'X222_DE_time', 'X169_DE_time', 'X105_DE_time','X238_DE_time', 'X209_DE_time',   'X135_DE_time']
# signal_labels = ['ball', 'inner', 'outer','ball2', 'inner2', 'outer2']
# file_names =['ball.mat', 'inner.mat', 'outer.mat','ball2.mat', 'inner2.mat', 'outer2.mat']
# data_columns = ['ball', 'iner', 'outer','ball2', 'inner2', 'outer2']
#######
# 读取信号数据
signals = []
for index in range(10):
    file_path = f'C:\\Users\\zh\\Desktop\\西储原始信号\\{file_names[index]}'
    data = loadmat(file_path)
    data_array = data[data_columns[index]].flatten()[:24000]
    signals.append(data_array)
# 计算最大振幅并动态确定间距
max_amp = max(np.max(np.abs(signal)) for signal in signals)
y_spacing = max_amp * 1
# 生成时间轴（假设采样频率为12kHz）
t = np.arange(24000)/12000  # 时间范围0-2秒

#######
# 创建3D图（调整画布尺寸为更适合论文排版的比例）
fig = plt.figure(figsize=(18, 12))  # Modified: 调整画布尺寸为更紧凑的比例
ax = fig.add_subplot(111, projection='3d')
ax.set_proj_type('ortho')

# 设置颜色循环（使用学术友好的颜色方案）
colors = plt.cm.tab10(np.linspace(0, 1, len(signals)))  # Modified: 添加颜色映射

# 绘制信号（优化线条参数）
for i, signal in enumerate(signals):
    ax.plot(t,
            signal,
            zs=i * y_spacing,
            zdir='y',
            color=colors[i],  # Modified: 添加颜色映射
            label=signal_labels[i],
            alpha=0.85,  # Modified: 调整透明度以平衡可读性
            linewidth=1.0)  # Modified: 适当加粗线条便于印刷显示
# Y轴标签设置（关键修改部分）
ax.set_yticks(np.arange(len(signal_labels)) * y_spacing)  # Modified: 动态生成刻度位置
ax.set_yticklabels(signal_labels,
                  rotation=45,          # Modified: 添加45度旋转防止重叠
                  ha='right',            # Modified: 水平对齐方式
                  va='center')          # Modified: 垂直对齐方式
# X轴设置（关键修改部分）
ax.set_xticks([0, 0.5, 1, 1.5, 2])  # Modified: 设置指定刻度位置
ax.set_xlim(0, 2.2)                 # Added: 确保显示完整刻度范围
# 坐标轴设置（优化标签间距和刻度显示）
ax.set_xlabel('Time (s)', fontsize=20, labelpad=15)  # Modified: 减少labelpad
ax.set_ylabel('', fontsize=20, labelpad=10)  # Modified: 调整标签间距
ax.set_zlabel('Amplitude', fontsize=20, labelpad=15)  # Modified: 修正拼写错误 & 调整间距

# 设置刻度参数（优化刻度显示）
ax.tick_params(axis='x', labelsize=20, pad=2,direction='out')  # Modified: 减少刻度标签间距
ax.tick_params(axis='y', labelsize=20, pad=15)
ax.tick_params(axis='z', labelsize=20, pad=10)
ax.tick_params(axis='both', direction='out')  # Modified: 设置刻度朝外

# 设置轴范围（保持原优化）
ax.set_ylim(-0.1 * y_spacing, (len(signals) - 0.9) * y_spacing)
ax.set_zlim(-max_amp * 1.1, max_amp * 1.1)

# 调整轴显示（优化3D轴显示）
ax.zaxis._axinfo['juggled'] = (1, 2, 0)
ax.xaxis.pane.set_edgecolor('k')  # Modified: 设置坐标平面边框颜色
ax.yaxis.pane.set_edgecolor('k')
ax.zaxis.pane.set_edgecolor('k')

# 设置网格线（添加细网格辅助观察）
ax.xaxis._axinfo["grid"].update({"linewidth": 0.1, "color": "gray"})  # Modified: 添加细网格
ax.yaxis._axinfo["grid"].update({"linewidth": 0.1, "color": "gray"})
ax.zaxis._axinfo["grid"].update({"linewidth": 0.1, "color": "gray"})

# 视角调整（优化观察角度）
ax.view_init(elev=22, azim=-35)  # Modified: 微调视角参数

# 图形美化（优化整体显示）
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.fill = False  # Modified: 保持坐标平面透明
    axis.pane.set_alpha(0.95)  # Modified: 添加轻微透明度
plt.tight_layout()  # Modified: 添加自动布局调整
plt.show()