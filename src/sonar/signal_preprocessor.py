#步骤3
#这一步骤是关键，需要等下更详细的处理
import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple, Optional
from signal_generator import generate_swept_sinusoid, generate_continuous_signal


def receive_reflect_signal(total_duration: float, fs: int = 48000) -> np.ndarray:
    # 计算精确的总样本数
    n_samples = int(total_duration * fs)

    # 生成环境噪声 (确保长度正确)
    ambient_noise = np.random.normal(0, 0.01, n_samples)

    # 生成反射信号
    single_sweep, _ = generate_swept_sinusoid()
    reflected_signal = generate_continuous_signal(
        single_sweep,
        total_duration=total_duration,
        fs=fs
    )

    # 强制对齐长度（核心修复）
    if len(reflected_signal) < n_samples:
        # 如果反射信号较短，补零
        reflected_signal = np.pad(reflected_signal, (0, n_samples - len(reflected_signal)), mode='constant')
    else:
        # 如果反射信号较长，截断
        reflected_signal = reflected_signal[:n_samples]

    # 混合信号
    received_signal = ambient_noise + 0.6 * reflected_signal
    return received_signal


def butter_bandpass_filter(
        signal: np.ndarray,
        lowcut: float = 18000.0,
        highcut: float = 22000.0,
        fs: int = 48000,
        order: int = 4
) -> np.ndarray:
    """
    带通滤波：保留18-22kHz声纳信号，过滤低频干扰（如人声、50/60Hz工频），参考论文🔶3-71🔶。
    """
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    # 设计Butterworth带通滤波器
    b, a = butter(order, [low, high], btype="band")
    # 零相位滤波（避免信号相位偏移）
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def preprocess_signal(
        received_signal: np.ndarray,
        transmit_signal: np.ndarray,
        fs: int = 48000
) -> np.ndarray:
    """
    信号预处理全流程，严格遵循论文步骤：
    1. 同步解调：基于相关性剔除直达信号（扬声器到麦克风的直接信号）
    2. 带通滤波：保留18-22kHz声纳频段
    3. 平滑降噪：10ms滑动窗口，抑制尖锐噪声（如关门声）
    """
    # 1. 同步解调：分离直达信号与反射信号
    # 取0.1s发射信号作为模板，计算相关性（识别直达信号区域）
    template = transmit_signal[:int(fs * 0.1)]
    corr = np.correlate(received_signal, template, mode="same")
    # 直达信号掩码：相关性>80%最大值的区域判定为直达信号
    direct_mask = corr > 0.8 * np.max(corr)
    # 剔除直达信号
    reflected_clean = received_signal.copy()
    reflected_clean[direct_mask] = 0

    # 2. 带通滤波：过滤18-22kHz外的噪声
    filtered_signal = butter_bandpass_filter(reflected_clean, fs=fs)

    # 3. 平滑降噪：10ms滑动窗口平均（参考论文趋势分析逻辑）
    window_size = int(fs * 0.01)  # 10ms窗口
    smoothed_signal = np.convolve(filtered_signal, np.ones(window_size) / window_size, mode="same")

    return smoothed_signal


if __name__ == "__main__":
    # 示例：接收并预处理10秒信号
    fs = 48000
    total_duration = 10.0

    # 1. 接收反射信号
    received_signal = receive_reflect_signal(total_duration=total_duration, fs=fs)

    # 2. 生成发射信号（用于同步解调）
    single_sweep, _ = generate_swept_sinusoid()
    transmit_signal = generate_continuous_signal(single_sweep, total_duration=total_duration, fs=fs)

    # 3. 预处理信号
    preprocessed_signal = preprocess_signal(received_signal, transmit_signal, fs=fs)
    print(f"预处理完成 | 信号长度：{len(preprocessed_signal)} | 信号均值：{np.mean(preprocessed_signal):.6f}")