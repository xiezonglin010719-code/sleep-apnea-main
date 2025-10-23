import numpy as np
from scipy.signal import stft, find_peaks, butter, filtfilt
from scipy.ndimage import morphological_gradient
from typing import Dict, Tuple
from signal_generator import generate_swept_sinusoid, generate_continuous_signal
from signal_preprocessor import receive_reflect_signal, preprocess_signal


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def calculate_frequency_shift(
        preprocessed_signal: np.ndarray,
        transmit_signal: np.ndarray,
        fs: int = 48000,
        n_fft: int = 1024,
        hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray, float]:
    """新增返回STFT实际采样率，解决参数混淆问题"""
    f_trans, t_trans, Zxx_trans = stft(transmit_signal, fs=fs, nperseg=n_fft, noverlap=hop_length)
    f_reflect, t_reflect, Zxx_reflect = stft(preprocessed_signal, fs=fs, nperseg=n_fft, noverlap=hop_length)

    min_time_len = min(len(t_trans), len(t_reflect))
    t = t_trans[:min_time_len]
    freq_shift = np.zeros(min_time_len)

    for i in range(min_time_len):
        f_trans_main = f_trans[np.argmax(np.abs(Zxx_trans[:, i]))]
        f_reflect_main = f_reflect[np.argmax(np.abs(Zxx_reflect[:, i]))]
        freq_shift[i] = f_reflect_main - f_trans_main

    # 计算STFT实际采样率（帧速率）
    fs_stft = 1 / np.mean(np.diff(t)) if len(t) > 1 else fs
    return freq_shift, t, fs_stft  # 新增返回fs_stft


def map_to_distance_change(
        freq_shift: np.ndarray,
        f_start: float = 18000.0,
        f_end: float = 22000.0,
        sweep_duration: float = 0.01075,
        sound_speed: float = 343.0
) -> np.ndarray:
    freq_band = f_end - f_start
    distance_change = (freq_shift * sound_speed * sweep_duration) / (2 * freq_band)
    return distance_change


def extract_breathing_features(
        distance_change: np.ndarray,
        time: np.ndarray,
        fs_stft: float,  # 改为STFT实际采样率（关键修正）
        lowpass_cutoff: float = 1.0  # 呼吸信号低频特性（0.1-1Hz）
) -> Dict:
    """
    核心修正点：
    1. 使用fs_stft（STFT采样率）替代原始fs，解决时间尺度错配
    2. 动态调整峰值阈值和间隔，适配实际呼吸周期
    3. 增加包络平滑度，减少噪声干扰
    """
    # 1. 低通滤波（使用正确的STFT采样率）
    distance_change_filtered = butter_lowpass_filter(
        distance_change,
        cutoff=lowpass_cutoff,
        fs=int(fs_stft)  # 修正：使用STFT采样率
    )

    # 2. 呼吸包络提取（增强平滑）
    envelope = np.abs(morphological_gradient(distance_change_filtered, size=(3,)))
    # 平滑窗口改为0.8秒（适应正常呼吸周期3-5秒）
    window_size = max(int(fs_stft * 0.8), 3)  # 窗口至少3个点
    envelope_smoothed = np.convolve(envelope, np.ones(window_size) / window_size, mode="same")

    # 3. 峰值检测参数优化（动态适配呼吸频率）
    # 阈值提高到均值+1.5倍标准差，过滤更多噪声
    peak_threshold = np.mean(envelope_smoothed) + 1.5 * np.std(envelope_smoothed)
    # 最小峰值间隔设为2秒（对应最大呼吸频率30次/分钟）
    min_peak_distance = max(int(fs_stft * 2.0), 5)  # 至少间隔5个点

    # 增加峰值 prominence 条件（确保峰值具有显著凸起）
    peaks, properties = find_peaks(
        envelope_smoothed,
        height=peak_threshold,
        distance=min_peak_distance,
        prominence=0.1 * np.max(envelope_smoothed)  # 突出度至少为最大值的10%
    )

    # 4. 呼吸周期过滤（严格范围校验）
    breath_periods = np.diff(time[peaks]) if len(peaks) > 1 else np.array([])
    breath_freq = 0.0
    valid_periods = np.array([])

    if len(breath_periods) > 0:
        # 正常呼吸周期范围：2-5秒（对应12-30次/分钟）
        valid_periods = breath_periods[(breath_periods >= 2.0) & (breath_periods <= 5.0)]
        if len(valid_periods) > 0:
            breath_freq = 60 / np.mean(valid_periods)
            # 进一步限制频率范围
            breath_freq = np.clip(breath_freq, 12.0, 30.0)

    return {
        "distance_change_raw": distance_change,
        "distance_change_filtered": distance_change_filtered,
        "breath_envelope": envelope_smoothed,
        "breath_peaks": peaks,
        "breath_peak_properties": properties,  # 新增峰值属性用于调试
        "breath_periods": breath_periods,
        "breath_periods_valid": valid_periods,
        "breath_freq": breath_freq,
        "time": time,
        "fs_stft": fs_stft  # 保存采样率用于后续处理
    }


if __name__ == "__main__":
    fs = 48000
    total_duration = 10.0

    # 生成测试信号
    single_sweep, _ = generate_swept_sinusoid()
    transmit_signal = generate_continuous_signal(single_sweep, total_duration=total_duration, fs=fs)
    received_signal = receive_reflect_signal(total_duration=total_duration, fs=fs)
    preprocessed_signal = preprocess_signal(received_signal, transmit_signal, fs=fs)

    # 计算频率偏移（获取STFT采样率）
    freq_shift, time, fs_stft = calculate_frequency_shift(preprocessed_signal, transmit_signal, fs=fs)
    distance_change = map_to_distance_change(freq_shift)

    # 提取特征（传入正确的STFT采样率）
    breathing_features = extract_breathing_features(distance_change, time, fs_stft=fs_stft)

    # 输出调试信息
    print(f"STFT采样率：{fs_stft:.1f}Hz | 原始信号采样率：{fs}Hz")
    print(f"呼吸频率：{breathing_features['breath_freq']:.1f}次/分钟")
    print(f"检测到的峰值数：{len(breathing_features['breath_peaks'])}")
    if len(breathing_features['breath_periods_valid']) > 0:
        print(
            f"有效周期数：{len(breathing_features['breath_periods_valid'])} | 平均周期：{np.mean(breathing_features['breath_periods_valid']):.2f}s")
    else:
        print("未检测到有效呼吸周期")