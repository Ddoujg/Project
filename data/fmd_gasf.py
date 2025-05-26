import os
import numpy as np
import h5py
from scipy.io import loadmat

# 配置参数
file_names = ['X118_DE_time', 'X185_DE_time', 'X222_DE_time', 'X105_DE_time',
              'X169_DE_time', 'X209_DE_time', 'X097_DE_time', 'X135_DE_time',
              'X201_DE_time', 'X238_DE_time']
data_column = 'y_final'  # 所有文件使用相同的列名


def normalization(data):
    """归一化数据到[-1, 1]范围"""
    _range = np.max(np.abs(data))
    return data / _range if _range != 0 else data


def optimized_gasf(data):
    """向量化实现的GASF转换"""
    x = normalization(data)
    theta = np.arccos(x)
    theta_sum = theta[:, None] + theta[None, :]  # 向量化加法
    return np.cos(theta_sum)


# 创建HDF5文件存储数据集
with h5py.File('train_dataset.h5', 'w') as h5_train, \
        h5py.File('val_dataset.h5', 'w') as h5_val:
    # 初始化数据集
    train_data = h5_train.create_dataset(
        'data',
        shape=(10 * 800, 3, 224, 224),
        dtype=np.float32,
        chunks=(100, 3, 224, 224)  # 分块存储优化IO
    )
    train_labels = h5_train.create_dataset(
        'labels',
        shape=(10 * 800,),
        dtype=np.int32
    )

    val_data = h5_val.create_dataset(
        'data',
        shape=(10 * 300, 3, 224, 224),
        dtype=np.float32,
        chunks=(100, 3, 224, 224)
    )
    val_labels = h5_val.create_dataset(
        'labels',
        shape=(10 * 300,),
        dtype=np.int32
    )

    # 处理每个文件
    for file_idx, file_name in enumerate(file_names):
        # 加载数据
        mat_data = loadmat(f'C:\\Users\\zh\\Desktop\\西储fmd信号\\{file_name}')
        raw_data = mat_data[data_column].T  # 形状 (3, N)

        # 初始化存储
        file_train = []
        file_val = []

        # 处理每个通道
        for channel in range(3):
            channel_data = raw_data[channel]

            # 生成训练样本
            train_segments = [
                optimized_gasf(channel_data[i:i + 224])
                for i in range(0, 800 * 100, 100)
            ]

            # 生成验证样本
            val_segments = [
                optimized_gasf(channel_data[i:i + 224])
                for i in range(80000, 80000 + 300 * 100, 100)
            ]

            file_train.append(train_segments)
            file_val.append(val_segments)

        # 重组为(样本数, 通道, 高, 宽)
        file_train = np.transpose(file_train, (1, 0, 2, 3))  # (800, 3, 224, 224)
        file_val = np.transpose(file_val, (1, 0, 2, 3))  # (300, 3, 224, 224)

        # 写入HDF5
        train_start = file_idx * 800
        train_end = (file_idx + 1) * 800
        train_data[train_start:train_end] = file_train
        train_labels[train_start:train_end] = file_idx

        val_start = file_idx * 300
        val_end = (file_idx + 1) * 300
        val_data[val_start:val_end] = file_val
        val_labels[val_start:val_end] = file_idx

print("数据处理完成，结果已保存为HDF5文件")