import h5py
import numpy as np
from scipy.io import loadmat

# 配置参数
file_names = ['X118_DE_time', 'X185_DE_time', 'X222_DE_time', 'X105_DE_time',
              'X169_DE_time', 'X209_DE_time', 'X097_DE_time', 'X135_DE_time',
              'X201_DE_time', 'X238_DE_time']
data_columns = ['y_final'] * 10

samples_config = {
    'train': {'samples': 800, 'stride': 100, 'start_pos': 0},
    'val': {'samples': 300, 'stride': 100, 'start_pos': 80000}
}

def generate_segments(data, mode):
    cfg = samples_config[mode]
    indices = cfg['start_pos'] + np.arange(cfg['samples']) * cfg['stride']

    segments = np.empty((cfg['samples'], 3, 224), dtype=np.float32)
    for i, idx in enumerate(indices):
        for ch in range(3):
            segments[i, ch] = data[ch, idx:idx + 224]
    return segments

# 创建HDF5文件
with h5py.File('fmd_train_dataset.h5', 'w') as hf_train, \
        h5py.File('fmd_val_dataset.h5', 'w') as hf_val:
    # 初始化数据集
    hf_train.create_dataset(
        'data',
        shape=(8000, 3, 224),
        dtype=np.float32,
        chunks=(100, 3, 224)
    )
    hf_train.create_dataset(
        'labels',
        shape=(8000,),
        dtype=np.int32
    )

    hf_val.create_dataset(
        'data',
        shape=(3000, 3, 224),
        dtype=np.float32,
        chunks=(100, 3, 224)
    )
    hf_val.create_dataset(
        'labels',
        shape=(3000,),
        dtype=np.int32
    )

    # 初始化数据和标签的指针
    train_data_ptr = 0
    val_data_ptr = 0

    # 处理每个文件
    for class_idx in range(10):
        mat_data = loadmat(f'C:\\Users\\zh\\Desktop\\西储fmd信号\\{file_names[class_idx]}')
        raw_data = mat_data[data_columns[class_idx]].T

        # 生成数据
        train_data = generate_segments(raw_data, 'train')
        val_data = generate_segments(raw_data, 'val')

        # 写入训练数据
        hf_train['data'][train_data_ptr:train_data_ptr + 800] = train_data
        hf_train['labels'][train_data_ptr:train_data_ptr + 800] = np.full(800, class_idx, dtype=np.int32)
        train_data_ptr += 800

        # 写入验证数据
        hf_val['data'][val_data_ptr:val_data_ptr + 300] = val_data
        hf_val['labels'][val_data_ptr:val_data_ptr + 300] = np.full(300, class_idx, dtype=np.int32)
        val_data_ptr += 300

print("处理完成！生成文件：")
print("- fmd_train_dataset.h5 包含训练数据（8000样本）")
print("- fmd_val_dataset.h5 包含验证数据（3000样本）")

# 检查HDF5文件结构
def inspect_h5(file_path):
    with h5py.File(file_path, 'r') as hf:
        print(f"\n检查文件: {file_path}")
        print("数据集键名:", list(hf.keys()))
        print("数据形状:", hf['data'].shape)
        print("标签形状:", hf['labels'].shape)
        print("标签分布:", np.unique(hf['labels'][:], return_counts=True))

inspect_h5('fmd_train_dataset.h5')
inspect_h5('fmd_val_dataset.h5')