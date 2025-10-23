#步骤6

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional



def plot_signal_waveform(time_raw, raw_signal, preprocessed_signal, save_path=None):
    """time_raw: 与原始信号长度匹配的时间轴"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 只绘制前1秒数据（避免10秒数据点过多卡顿）
    plot_len = int(len(time_raw) * 0.1)  # 10%的数据量
    axes[0].plot(time_raw[:plot_len], raw_signal[:plot_len], color="#1f77b4", alpha=0.8)
    axes[0].set_title("Raw Signal (with Direct Path & Noise)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_raw[:plot_len], preprocessed_signal[:plot_len], color="#ff7f0e", alpha=0.8)
    axes[1].set_title("Preprocessed Signal")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_breathing_features(breathing_features, save_path=None):
    time = breathing_features["time"]
    distance = breathing_features["distance_change"]
    envelope = breathing_features["breath_envelope"]
    peaks = breathing_features["breath_peaks"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, distance, color="#2ca02c", alpha=0.6, label="Distance Change (m)")
    ax.plot(time, envelope, color="#d62728", linewidth=2, label="Breath Envelope")
    ax.scatter(time[peaks], envelope[peaks], color="#9467bd", s=50, label="Breath Peaks")

    ax.set_title(f"Breathing Features (Frequency: {breathing_features['breath_freq']:.1f} breaths/min)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_apnea_events(breathing_features, events, save_path=None):
    time = breathing_features["time"]
    envelope = breathing_features["breath_envelope"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, envelope, color="#1f77b4", linewidth=2, label="Breath Envelope")

    for event in events:
        color = "#d62728" if event["event_type"] == "apnea" else "#ff7f0e"
        ax.axvspan(event["start_time"], event["end_time"], alpha=0.3, color=color,
                   label=f"{event['event_type'].capitalize()}")

    # 去重图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_title("Breath Envelope with Events")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
if __name__ == "__main__":
    # 示例：可视化10秒信号的全流程结果
    from signal_generator import generate_swept_sinusoid, generate_continuous_signal
    from signal_preprocessor import receive_reflect_signal, preprocess_signal
    from feature_extractor import calculate_frequency_shift, map_to_distance_change, extract_breathing_features
    from apnea_detector import detect_apnea_events

    fs = 48000
    total_duration = 10.0

    # 加载数据（增加异常捕获和长度验证）
    try:
        single_sweep, _ = generate_swept_sinusoid()
        transmit_signal = generate_continuous_signal(single_sweep, total_duration=total_duration, fs=fs)
        received_signal = receive_reflect_signal(total_duration=total_duration, fs=fs)
        preprocessed_signal = preprocess_signal(received_signal, transmit_signal, fs=fs)

        # 验证原始信号长度
        assert len(received_signal) == int(
            fs * total_duration), f"原始信号长度错误：{len(received_signal)} != {int(fs * total_duration)}"
        assert len(preprocessed_signal) == int(fs * total_duration), f"预处理信号长度错误"

        # 计算频率偏移和距离变化
        freq_shift, stft_time = calculate_frequency_shift(preprocessed_signal, transmit_signal, fs=fs)
        distance_change = map_to_distance_change(freq_shift)

        # 验证STFT相关数据长度
        assert len(freq_shift) == len(stft_time), f"频率偏移与STFT时间轴长度不匹配"
        assert len(distance_change) == len(stft_time), f"距离变化与STFT时间轴长度不匹配"

        # 生成样本级时间轴（用于原始信号绘图）
        sample_time = np.linspace(0, total_duration, len(received_signal))

        # 生成呼吸特征
        breathing_features = extract_breathing_features(distance_change, stft_time, fs=fs)
        apnea_events = detect_apnea_events(breathing_features)

        # 可视化（增加数据非空判断）
        if len(sample_time) > 0 and len(received_signal) > 0 and len(preprocessed_signal) > 0:
            plot_signal_waveform(sample_time, received_signal, preprocessed_signal, save_path="signal_waveform.png")
        else:
            print("原始信号数据为空，无法绘制信号波形图")

        if len(breathing_features["time"]) > 0 and len(breathing_features["distance_change"]) > 0:
            plot_breathing_features(breathing_features, save_path="breathing_features.png")
        else:
            print("呼吸特征数据为空，无法绘制呼吸特征图")

        if len(breathing_features["time"]) > 0 and len(apnea_events) >= 0:
            plot_apnea_events(breathing_features, apnea_events, save_path="apnea_events.png")
        else:
            print("事件数据异常，无法绘制事件图")

    except Exception as e:
        print(f"数据加载或可视化失败：{str(e)}")