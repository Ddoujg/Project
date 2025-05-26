import h5py
import numpy as np
from scipy.io import loadmat

# 配置参数
file_names = ['X118_DE_time', 'X185_DE_time', 'X222_DE_time', 'X105_DE_time',
              'X169_DE_time', 'X209_DE_time', 'X097_DE_time', 'X135_DE_time',
              'X201_DE_time', 'X238_DE_time']
data_columns = file_names  # 假设MAT文件中的变量名与文件名一致

samples_config = {
    'train': {'samples_per_class': 800, 'stride': 100, 'start_pos': 0},
    'val': {'samples_per_class': 300, 'stride': 100, 'start_pos': 80000}
}


def process_data(file_path, column_name, target, mode):
    """处理单个文件的数据"""
    # 加载MAT数据
    mat_data = loadmat(file_path)
    raw_data = mat_data[column_name].flatten().astype(np.float32)

    # 计算参数
    cfg = samples_config[mode]
    total_required = cfg['start_pos'] + (cfg['samples_per_class'] - 1) * cfg['stride'] + 224

    # 数据长度验证
    if len(raw_data) < total_required:
        raise ValueError(f"{file_path} 数据长度不足! 需要{total_required}, 实际{len(raw_data)}")

    # 生成索引
    indices = cfg['start_pos'] + np.arange(cfg['samples_per_class']) * cfg['stride']

    # 预分配内存并处理
    segments = np.empty((cfg['samples_per_class'], 1, 224), dtype=np.float32)
    for i, idx in enumerate(indices):
        window = raw_data[idx:idx + 224]
        segments[i, 0, :] = window / np.max(np.abs(window))  # 窗口级归一化

    return segments


# 创建HDF5文件
with h5py.File('train_data.h5', 'w') as hf_train, \
        h5py.File('val_data.h5', 'w') as hf_val:
    # 初始化数据集
    for mode, hf in [('train', hf_train), ('val', hf_val)]:
        samples = samples_config[mode]['samples_per_class'] * len(file_names)
        hf.create_dataset(
            'data',
            shape=(samples, 1, 224),
            dtype=np.float32,
            chunks=(1000, 1, 224)  # 分块优化
        )
        hf.create_dataset(
            'labels',
            shape=(samples,),
            dtype=np.int32
        )

    # 处理每个文件
    for class_idx, (file_name, column) in enumerate(zip(file_names, data_columns)):
        # 训练数据处理
        train_segments = process_data(
            f'C:\\Users\\zh\\Desktop\\西储原始信号\\{file_name}',
            column,
            target='train',
            mode='train'
        )

        # 验证数据处理
        val_segments = process_data(
            f'C:\\Users\\zh\\Desktop\\西储原始信号\\{file_name}',
            column,
            target='val',
            mode='val'
        )

        # 计算存储位置
        train_start = class_idx * 800
        val_start = class_idx * 300

        # 写入训练数据
        hf_train['data'][train_start:train_start + 800] = train_segments
        hf_train['labels'][train_start:train_start + 800] = class_idx

        # 写入验证数据
        hf_val['data'][val_start:val_start + 300] = val_segments
        hf_val['labels'][val_start:val_start + 300] = class_idx

print("处理完成！生成文件：")
print("- train_data.h5 包含训练数据（8000样本）")
print("- val_data.h5 包含验证数据（3000样本）")