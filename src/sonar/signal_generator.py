#步骤1
import numpy as np
from typing import Tuple


def generate_swept_sinusoid(
        f_start: float = 18000.0,  # 起始频率（Hz，论文推荐18kHz）
        f_end: float = 22000.0,  # 结束频率（Hz，论文推荐22kHz）
        sweep_duration: float = 0.01075,  # 扫频周期（s，论文10.75ms）
        fs: int = 48000  # 采样率（Hz，论文统一48kHz）
) -> Tuple[np.ndarray, int]:
    """
    生成线性扫频正弦波信号，参考论文公式：
    1. 瞬时频率：f(t) = f_start + (f_end - f_start) * t / sweep_duration
    2. 信号相位：φ(t) = 2π ∫₀ᵗ f(τ)dτ（避免相位不连续）
    3. 时域信号：s(t) = sin(φ(t))
    """
    # 生成时间序列（覆盖单个扫频周期）
    t = np.linspace(0, sweep_duration, int(fs * sweep_duration), endpoint=False)
    # 计算瞬时频率（线性变化）
    instant_freq = f_start + (f_end - f_start) * t / sweep_duration
    # 积分计算相位（确保信号连续性）
    phase = 2 * np.pi * np.cumsum(instant_freq) / fs
    # 生成扫频信号并归一化（适配扬声器输出范围[-1,1]）
    swept_signal = np.sin(phase)
    swept_signal = swept_signal / np.max(np.abs(swept_signal))

    return swept_signal, fs


def generate_continuous_signal(
        swept_signal: np.ndarray,
        total_duration: float = 10.0,  # 总信号时长（s，论文样本标准时长）
        fs: int = 48000
) -> np.ndarray:
    """
    生成连续扫频信号（重复单个扫频周期），参考论文：
    - 多周期连续发射，确保覆盖完整呼吸监测片段
    - 总时长可灵活调整（支持多晚监测需求）
    """
    # 单个扫频周期的样本数
    single_sweep_len = len(swept_signal)
    # 计算总周期数（确保总时长达标）
    total_cycles = int(total_duration * fs / single_sweep_len)
    # 拼接连续信号
    continuous_signal = np.tile(swept_signal, total_cycles)
    # 截取到目标总时长（避免多余数据）
    continuous_signal = continuous_signal[:int(total_duration * fs)]

    return continuous_signal


if __name__ == "__main__":
    # 示例：生成10秒连续扫频信号（论文典型样本时长）
    single_sweep, fs = generate_swept_sinusoid()
    continuous_sweep = generate_continuous_signal(single_sweep, total_duration=10.0, fs=fs)
    print(f"信号生成完成 | 时长：10s | 采样率：{fs}Hz | 数据长度：{len(continuous_sweep)}")
    print(f"信号频段：{18000}-{22000}Hz（超出成人听觉范围，无干扰）")