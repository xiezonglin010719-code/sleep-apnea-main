# import numpy as np
# import os
# from typing import Optional, List, Dict
#
# from signal_generator import generate_swept_sinusoid, generate_continuous_signal
# from signal_transmitter import get_speaker_info, transmit_signal
# from signal_preprocessor import receive_reflect_signal, preprocess_signal
# from feature_extractor import calculate_frequency_shift, map_to_distance_change, extract_breathing_features
# from apnea_detector import detect_apnea_events
# from result_visualizer import plot_signal_waveform, plot_breathing_features, plot_apnea_events
#
#
# def calculate_ahi(
#         apnea_events: List[Dict],
#         total_sleep_time: float = 3600.0  # 总睡眠时间（s，默认1小时）
# ) -> float:
#     """估算AHI（参考论文：事件数/总睡眠时间（小时））"""
#     total_events = len([e for e in apnea_events if e["event_type"] in ["apnea", "hypopnea"]])
#     ahi = total_events / (total_sleep_time / 3600)
#     return round(ahi, 1)
#
#
# def generate_detection_report(
#         apnea_events: List[Dict],
#         breathing_features: Dict,
#         total_sleep_time: float = 3600.0
# ) -> str:
#     """生成检测报告（符合论文输出格式）"""
#     ahi = calculate_ahi(apnea_events, total_sleep_time)
#     breath_freq = breathing_features["breath_freq"]
#     apnea_count = len([e for e in apnea_events if e["event_type"] == "apnea"])
#     hypopnea_count = len([e for e in apnea_events if e["event_type"] == "hypopnea"])
#
#     # OSA严重程度判定（参考论文临床阈值）
#     if ahi < 5:
#         osa_severity = "Normal (无OSA)"
#     elif 5 <= ahi < 15:
#         osa_severity = "Mild OSA (轻度OSA)"
#     elif 15 <= ahi < 30:
#         osa_severity = "Moderate OSA (中度OSA)"
#     else:
#         osa_severity = "Severe OSA (重度OSA)"
#
#     report = f"""
# # 睡眠呼吸暂停检测报告（基于主动声纳技术）
# ## 1. 检测参数
# - 检测时长：{total_sleep_time / 3600:.1f}小时
# - 声纳信号频段：18-22kHz（超出成人听觉范围）
# - 采样率：48kHz
#
# ## 2. 核心指标
# - 呼吸频率：{breath_freq:.1f}次/分钟
# - 呼吸暂停事件数：{apnea_count}个
# - 低通气事件数：{hypopnea_count}个
# - 呼吸暂停低通气指数（AHI）：{ahi}次/小时
#
# ## 3. OSA严重程度判定
# - 结果：{osa_severity}
# - 建议：中度/重度OSA需进一步PSG确认，支持多晚监测提高可靠性
#
# ## 4. 事件详情
# """
#     report += "- 未检测到事件\n" if len(apnea_events) == 0 else ""
#     for idx, e in enumerate(apnea_events, 1):
#         report += f"- 事件{idx}：类型={e['event_type']}，起始={e['start_time']:.1f}s，持续={e['duration']:.1f}s\n"
#     return report
#
#
# def main(
#         total_duration: float = 10.0,
#         visualize: bool = True,
#         save_visual: bool = True,
#         save_report: bool = True
# ) -> None:
#     """主程序：端到端检测流程"""
#     print("=" * 60)
#     print("睡眠呼吸暂停检测系统（基于jtd-12-08-4476.pdf主动声纳技术）")
#     print("=" * 60)
#
#     # 结果目录创建
#     result_dir = "detection_results"
#     os.makedirs(result_dir, exist_ok=True)
#
#     # 1. 信号生成
#     print("\n【1/7】生成18-22kHz扫频信号...")
#     single_sweep, fs = generate_swept_sinusoid()
#     continuous_signal = generate_continuous_signal(single_sweep, total_duration=total_duration, fs=fs)
#
#     # 2. 信号发射
#     print("\n【2/7】发射声纳信号...")
#     speaker_info = get_speaker_info()
#     transmit_signal(continuous_signal, fs=fs, device_id=speaker_info["device_id"])
#
#     # 3. 信号接收
#     print("\n【3/7】接收反射信号...")
#     received_signal = receive_reflect_signal(total_duration=total_duration, fs=fs)
#
#     # 4. 信号预处理
#     print("\n【4/7】预处理（剔除直达信号+滤波+降噪）...")
#     preprocessed_signal = preprocess_signal(received_signal, continuous_signal, fs=fs)
#
#     # 步骤5. 特征提取（原代码保留）
#     print("\n【5/7】提取呼吸特征...")
#     freq_shift, time_stft = calculate_frequency_shift(preprocessed_signal, continuous_signal, fs=fs)  # 重命名为time_stft
#     distance_change = map_to_distance_change(freq_shift)
#     breathing_features = extract_breathing_features(distance_change, time_stft, fs=fs)  # 使用STFT时间轴
#
#     # 新增：为原始信号和预处理信号生成匹配的时间轴
#     signal_duration = len(received_signal) / fs  # 实际信号时长
#     time_raw = np.linspace(0, signal_duration, len(received_signal), endpoint=False)  # 与原始信号长度一致
#
#     # 步骤6. 事件识别（原代码保留）
#     print("\n【6/7】识别呼吸事件...")
#     apnea_events = detect_apnea_events(breathing_features)
#
#     # 步骤7. 结果输出（修改可视化部分的time参数）
#     if visualize:
#         print("\n生成可视化结果...")
#         # 传入原始信号对应的时间轴time_raw，而非STFT的time_stft
#         plot_signal_waveform(
#             time_raw,  # 修正：使用原始信号时间轴
#             received_signal,
#             preprocessed_signal,
#             save_path=os.path.join(result_dir, "signal.png") if save_visual else None
#         )
#         # 呼吸特征和事件图仍使用STFT时间轴（与特征长度匹配）
#         plot_breathing_features(
#             breathing_features,
#             save_path=os.path.join(result_dir, "features.png") if save_visual else None
#         )
#         plot_apnea_events(
#             breathing_features,
#             apnea_events,
#             save_path=os.path.join(result_dir, "events.png") if save_visual else None
#         )
#
#
#
#
# if __name__ == "__main__":
#     main(
#         total_duration=10.0,  # 测试用10秒，实际建议≥3600秒（1小时）
#         visualize=True,
#         save_visual=True,
#         save_report=True
#     )


import numpy as np
import os
from typing import Optional, List, Dict

from signal_generator import generate_swept_sinusoid, generate_continuous_signal
from signal_transmitter import get_speaker_info, transmit_signal
from signal_preprocessor import receive_reflect_signal, preprocess_signal
from feature_extractor import calculate_frequency_shift, map_to_distance_change, extract_breathing_features
from apnea_detector import detect_apnea_events
from result_visualizer import plot_signal_waveform, plot_breathing_features, plot_apnea_events


def calculate_ahi(
        apnea_events: List[Dict],
        total_sleep_time: float = 3600.0  # 总睡眠时间（s，默认1小时）
) -> float:
    total_events = len([e for e in apnea_events if e["event_type"] in ["apnea", "hypopnea"]])
    ahi = total_events / (total_sleep_time / 3600)
    return round(ahi, 1)


def generate_detection_report(
        apnea_events: List[Dict],
        breathing_features: Dict,
        total_sleep_time: float = 3600.0
) -> str:
    ahi = calculate_ahi(apnea_events, total_sleep_time)
    breath_freq = breathing_features["breath_freq"]
    apnea_count = len([e for e in apnea_events if e["event_type"] == "apnea"])
    hypopnea_count = len([e for e in apnea_events if e["event_type"] == "hypopnea"])

    if ahi < 5:
        osa_severity = "Normal (无OSA)"
    elif 5 <= ahi < 15:
        osa_severity = "Mild OSA (轻度OSA)"
    elif 15 <= ahi < 30:
        osa_severity = "Moderate OSA (中度OSA)"
    else:
        osa_severity = "Severe OSA (重度OSA)"

    report = f"""
# 睡眠呼吸暂停检测报告（基于主动声纳技术）
## 1. 检测参数
- 检测时长：{total_sleep_time / 3600:.1f}小时
- 声纳信号频段：18-22kHz（超出成人听觉范围）
- 采样率：48kHz

## 2. 核心指标
- 呼吸频率：{breath_freq:.1f}次/分钟
- 呼吸暂停事件数：{apnea_count}个
- 低通气事件数：{hypopnea_count}个
- 呼吸暂停低通气指数（AHI）：{ahi}次/小时

## 3. OSA严重程度判定
- 结果：{osa_severity}
- 建议：中度/重度OSA需进一步PSG确认，支持多晚监测提高可靠性

## 4. 事件详情
"""
    report += "- 未检测到事件\n" if len(apnea_events) == 0 else ""
    for idx, e in enumerate(apnea_events, 1):
        report += f"- 事件{idx}：类型={e['event_type']}，起始={e['start_time']:.1f}s，持续={e['duration']:.1f}s\n"
    return report


def generate_mock_data(total_duration: float, fs: int) -> Dict:
    """生成模拟数据（模拟正常呼吸+1次呼吸暂停+1次低通气）"""
    # 1. 模拟原始信号（扫频信号+反射信号+噪声）
    t_raw = np.linspace(0, total_duration, int(fs * total_duration), endpoint=False)
    # 基础扫频信号（18-22kHz）
    f_start, f_end = 18000, 22000
    sweep = np.sin(2 * np.pi * (f_start * t_raw + 0.5 * (f_end - f_start) * t_raw**2 / total_duration))
    # 反射信号（模拟呼吸引起的幅度变化）
    breath_rate = 0.25  # 0.25Hz ≈ 15次/分钟
    breath_amplitude = 0.5 + 0.3 * np.sin(2 * np.pi * breath_rate * t_raw)
    # 加入呼吸暂停（3-6秒）和低通气（7-9秒）
    breath_amplitude[(t_raw >= 3) & (t_raw <= 6)] *= 0.1  # 暂停（幅度骤降90%）
    breath_amplitude[(t_raw >= 7) & (t_raw <= 9)] *= 0.5  # 低通气（幅度骤降50%）
    reflected = 0.6 * sweep * breath_amplitude
    # 原始接收信号 = 直达信号 + 反射信号 + 噪声
    received = sweep + reflected + 0.1 * np.random.randn(len(t_raw))
    preprocessed = reflected + 0.05 * np.random.randn(len(t_raw))  # 预处理后信号（去除直达波）

    # 2. 模拟频率偏移和距离变化
    stft_hop = 512
    stft_n_fft = 1024
    stft_time_len = len(t_raw) // stft_hop  # STFT时间轴长度
    time_stft = np.linspace(0, total_duration, stft_time_len, endpoint=False)
    # 频率偏移（与呼吸同步的低频变化）
    freq_shift = 50 * np.sin(2 * np.pi * breath_rate * time_stft)
    # 模拟暂停/低通气时的频率偏移变化
    freq_shift[(time_stft >= 3) & (time_stft <= 6)] *= 0.1  # 暂停时偏移减小
    freq_shift[(time_stft >= 7) & (time_stft <= 9)] *= 0.5  # 低通气时偏移减小
    # 距离变化（根据公式转换）
    distance_change = (freq_shift * 343 * 0.01075) / (2 * 4000)  # 4000=22000-18000

    # 3. 模拟呼吸特征
    # 生成呼吸包络（放大距离变化信号）
    envelope = np.abs(distance_change) * 1000  # 放大到可视化尺度
    # 呼吸峰值（正常呼吸段的峰值）
    normal_peaks = np.where((time_stft >= 0) & (time_stft < 3) |
                           (time_stft >= 6) & (time_stft < 7) |
                           (time_stft >= 9))[0]
    # 每隔3个点取一个峰值（模拟真实峰值分布）
    peaks = normal_peaks[::3]
    # 呼吸周期（约4秒）
    breath_periods = np.array([4.0, 4.1, 3.9])
    breath_freq = 60 / np.mean(breath_periods) if len(breath_periods) > 0 else 15.0

    breathing_features = {
        "distance_change": distance_change,
        "distance_change_filtered": distance_change,
        "breath_envelope": envelope,
        "breath_peaks": peaks,
        "breath_periods": breath_periods,
        "breath_periods_valid": breath_periods,
        "breath_freq": breath_freq,
        "time": time_stft
    }

    # 4. 模拟事件（与信号特征匹配）
    apnea_events = [
        {
            "event_type": "apnea",
            "start_time": 3.0,
            "end_time": 6.0,
            "duration": 3.0,
            "detection_method": "rule_based"
        },
        {
            "event_type": "hypopnea",
            "start_time": 7.0,
            "end_time": 9.0,
            "duration": 2.0,
            "detection_method": "rule_based"
        }
    ]

    return {
        "received_signal": received,
        "preprocessed_signal": preprocessed,
        "time_raw": t_raw,
        "time_stft": time_stft,
        "distance_change": distance_change,
        "breathing_features": breathing_features,
        "apnea_events": apnea_events
    }


def main(
        total_duration: float = 10.0,
        visualize: bool = True,
        save_visual: bool = True,
        save_report: bool = True,
        use_mock_data: bool = True  # 新增：是否使用模拟数据
) -> None:
    """主程序：端到端检测流程（支持模拟数据）"""
    print("=" * 60)
    print("睡眠呼吸暂停检测系统（基于jtd-12-08-4476.pdf主动声纳技术）")
    print("=" * 60)

    # 结果目录创建
    result_dir = "detection_results"
    os.makedirs(result_dir, exist_ok=True)

    # 生成或加载数据
    if use_mock_data:
        print("\n【使用模拟数据测试】")
        fs = 48000
        mock_data = generate_mock_data(total_duration, fs)
        received_signal = mock_data["received_signal"]
        preprocessed_signal = mock_data["preprocessed_signal"]
        time_raw = mock_data["time_raw"]
        time_stft = mock_data["time_stft"]
        distance_change = mock_data["distance_change"]
        breathing_features = mock_data["breathing_features"]
        apnea_events = mock_data["apnea_events"]
    else:
        # 1. 信号生成
        print("\n【1/7】生成18-22kHz扫频信号...")
        single_sweep, fs = generate_swept_sinusoid()
        continuous_signal = generate_continuous_signal(single_sweep, total_duration=total_duration, fs=fs)

        # 2. 信号发射
        print("\n【2/7】发射声纳信号...")
        speaker_info = get_speaker_info()
        transmit_signal(continuous_signal, fs=fs, device_id=speaker_info["device_id"])

        # 3. 信号接收
        print("\n【3/7】接收反射信号...")
        received_signal = receive_reflect_signal(total_duration=total_duration, fs=fs)

        # 4. 信号预处理
        print("\n【4/7】预处理（剔除直达信号+滤波+降噪）...")
        preprocessed_signal = preprocess_signal(received_signal, continuous_signal, fs=fs)

        # 5. 特征提取
        print("\n【5/7】提取呼吸特征...")
        freq_shift, time_stft = calculate_frequency_shift(preprocessed_signal, continuous_signal, fs=fs)
        distance_change = map_to_distance_change(freq_shift)
        breathing_features = extract_breathing_features(distance_change, time_stft, fs=fs)

        # 6. 生成原始信号时间轴
        signal_duration = len(received_signal) / fs
        time_raw = np.linspace(0, signal_duration, len(received_signal), endpoint=False)

        # 7. 事件识别
        print("\n【6/7】识别呼吸事件...")
        apnea_events = detect_apnea_events(breathing_features)

    # 8. 结果可视化
    if visualize:
        print("\n生成可视化结果...")
        plot_signal_waveform(
            time_raw,
            received_signal,
            preprocessed_signal,
            save_path=os.path.join(result_dir, "signal.png") if save_visual else None
        )
        plot_breathing_features(
            breathing_features,
            save_path=os.path.join(result_dir, "features.png") if save_visual else None
        )
        plot_apnea_events(
            breathing_features,
            apnea_events,
            save_path=os.path.join(result_dir, "events.png") if save_visual else None
        )

    # 9. 生成报告
    if save_report:
        report = generate_detection_report(apnea_events, breathing_features, total_duration)
        report_path = os.path.join(result_dir, "detection_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n检测报告已保存至：{report_path}")


if __name__ == "__main__":
    main(
        total_duration=10.0,
        visualize=True,
        save_visual=True,
        save_report=True,
        use_mock_data=True  # 设置为True使用模拟数据，False使用真实流程
    )