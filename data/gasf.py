import h5py
import numpy as np
from scipy.io import loadmat

# 配置参数
file_names = ['X118_DE_time', 'X185_DE_time', 'X222_DE_time', 'X105_DE_time',
              'X169_DE_time', 'X209_DE_time', 'X097_DE_time', 'X135_DE_time',
              'X201_DE_time', 'X238_DE_time']
data_columns = file_names  # 假设MAT文件中的键名与文件名一致

samples_config = {
    'train': {'samples': 800, 'stride': 100, 'start_pos': 0},
    'val': {'samples': 300, 'stride': 100, 'start_pos': 80000}
}


def vectorized_gasf(data):
    """向量化GASF计算"""
    x = data / np.max(np.abs(data))  # 归一化
    theta = np.arccos(x)
    # 使用广播计算角度和
    theta_sum = theta[:, None] + theta[None, :]
    return np.cos(theta_sum)


def process_file(file_path, column, mode):
    """处理单个文件生成GASF"""
    # 加载数据
    mat_data = loadmat(file_path)
    raw_data = mat_data[column].flatten().astype(np.float32)

    cfg = samples_config[mode]
    required_length = cfg['start_pos'] + (cfg['samples'] - 1) * cfg['stride'] + 224

    # 验证数据长度
    if len(raw_data) < required_length:
        raise ValueError(f"{file_path} 需要至少{required_length}采样点")

    # 预分配内存
    gasf_images = np.empty((cfg['samples'], 224, 224), dtype=np.float32)

    # 生成GASF
    for i in range(cfg['samples']):
        start = cfg['start_pos'] + i * cfg['stride']
        segment = raw_data[start:start + 224]
        gasf_images[i] = vectorized_gasf(segment)

    return gasf_images


# 创建HDF5文件
with h5py.File('train_gasf.h5', 'w') as hf_train, \
        h5py.File('val_gasf.h5', 'w') as hf_val:
    # 初始化训练数据集
    train_data = hf_train.create_dataset(
        'data',
        shape=(10 * 800, 1, 224, 224),
        dtype=np.float32,
        chunks=(500, 1, 224, 224)  # 分块存储
    )
    train_labels = hf_train.create_dataset(
        'labels',
        shape=(10 * 800,),
        dtype=np.int32
    )

    # 初始化验证数据集
    val_data = hf_val.create_dataset(
        'data',
        shape=(10 * 300, 1, 224, 224),
        dtype=np.float32,
        chunks=(300, 1, 224, 224)
    )
    val_labels = hf_val.create_dataset(
        'labels',
        shape=(10 * 300,),
        dtype=np.int32
    )

    # 处理每个文件
    for class_idx, (file_name, column) in enumerate(zip(file_names, data_columns)):
        # 生成训练数据
        train_gasf = process_file(
            f'C:\\Users\\zh\\Desktop\\西储原始信号\\{file_name}',
            column,
            'train'
        )

        # 生成验证数据
        val_gasf = process_file(
            f'C:\\Users\\zh\\Desktop\\西储原始信号\\{file_name}',
            column,
            'val'
        )

        # 计算存储位置
        train_start = class_idx * 800
        val_start = class_idx * 300

        # 写入训练数据
        train_data[train_start:train_start + 800] = train_gasf[:, None, :, :]  # 添加通道维度
        train_labels[train_start:train_start + 800] = class_idx

        # 写入验证数据
        val_data[val_start:val_start + 300] = val_gasf[:, None, :, :]
        val_labels[val_start:val_start + 300] = class_idx

print("处理完成！生成文件：")
print("- train_gasf.h5 (训练数据: 8000样本)")
print("- val_gasf.h5 (验证数据: 3000样本)")