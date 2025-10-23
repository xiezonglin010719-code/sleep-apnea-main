import numpy as np
import pickle


def to_uint8_image(signal):
    """将音频特征（如梅尔频谱）归一化并转换为uint8格式"""
    # 归一化到[0, 1]
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    if signal_max - signal_min < 1e-6:  # 避免除以0
        normalized = np.zeros_like(signal)
    else:
        normalized = (signal - signal_min) / (signal_max - signal_min)
    # 转换为uint8（0-255）
    return (normalized * 255).astype(np.uint8)


def load_pickle_events(pickle_path):
    """加载pickle文件中的事件数据"""
    with open(pickle_path, 'rb') as f:
        events = pickle.load(f)
    return events