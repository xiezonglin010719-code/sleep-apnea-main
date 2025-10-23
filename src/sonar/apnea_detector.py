#模块5
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional, Tuple
from feature_extractor import calculate_frequency_shift, map_to_distance_change, extract_breathing_features
from signal_generator import generate_swept_sinusoid, generate_continuous_signal
from signal_preprocessor import receive_reflect_signal, preprocess_signal


def extract_event_features(
        breathing_features: Dict,
        window_size: int = 5  # 特征窗口大小（s，参考论文25-100s时间尺度）
) -> np.ndarray:
    """
    提取事件分类特征（用于逻辑回归模型），参考论文：
    特征包括幅度统计量、幅度骤降比例、峰值密度，覆盖呼吸事件的核心特征
    """
    envelope = breathing_features["breath_envelope"]
    time = breathing_features["time"]
    fs = len(envelope) / time[-1]
    window_samples = int(window_size * fs)  # 窗口对应的样本数

    event_features = []
    # 滑动窗口提取特征
    for i in range(0, len(envelope) - window_samples, window_samples):
        window = envelope[i:i + window_samples]
        # 特征1：窗口内幅度均值（反映呼吸深度）
        mean_amp = np.mean(window)
        # 特征2：幅度标准差（反映呼吸稳定性）
        std_amp = np.std(window)
        # 特征3：幅度骤降比例（低于最大幅度70%的样本占比，参考论文30%骤降阈值）
        amp_max = np.max(envelope)
        drop_ratio = np.sum(window < 0.7 * amp_max) / len(window)
        # 特征4：峰值密度（窗口内呼吸峰值数/窗口时长，反映呼吸频率稳定性）
        peaks_in_window = [p for p in breathing_features["breath_peaks"] if i <= p < i + window_samples]
        peak_density = len(peaks_in_window) / window_size

        event_features.append([mean_amp, std_amp, drop_ratio, peak_density])

    return np.array(event_features)


def train_apnea_model(
        X_train: np.ndarray,
        y_train: np.ndarray
) -> Tuple[LogisticRegression, StandardScaler]:
    """
    训练呼吸暂停分类模型（逻辑回归），参考论文：
    - 采用五折交叉验证优化正则化参数（此处简化为基础训练）
    - 类别权重平衡，应对样本不平衡问题（如正常呼吸样本多于事件样本）
    """
    # 特征标准化（提升模型收敛性）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 训练逻辑回归模型
    model = LogisticRegression(
        class_weight="balanced",  # 平衡类别权重
        max_iter=1000,  # 增加迭代次数，确保收敛
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler


def detect_apnea_events(
        breathing_features: Dict,
        model: Optional[LogisticRegression] = None,
        scaler: Optional[StandardScaler] = None,
        amp_drop_threshold: float = 0.3,  # 幅度骤降阈值（30%，论文临床标准）
        duration_threshold: float = 10.0  # 事件持续阈值（10秒，论文临床标准）
) -> List[Dict]:
    """
    识别呼吸暂停/低通气事件，参考论文规则：
    1. 呼吸暂停：幅度骤降≥30% + 持续≥10秒 + 无呼吸峰值（无呼吸运动）
    2. 低通气：幅度骤降≥30% + 持续≥10秒 + 有呼吸峰值（呼吸运动减弱）
    3. 支持模型预测（逻辑回归）与规则检测双模式
    """
    envelope = breathing_features["breath_envelope"]
    peaks = breathing_features["breath_peaks"]
    # 修正键访问错误：直接访问"time"键，而非嵌套的"breath_features"
    time = breathing_features["time"]
    fs = len(envelope) / time[-1]
    amp_max = np.max(envelope)
    amp_threshold = amp_max * (1 - amp_drop_threshold)  # 幅度骤降阈值（70%最大幅度）

    apnea_events = []
    low_amp_start = None  # 低幅度片段起始索引

    # 1. 基于规则的事件检测（无模型时默认使用）
    for i in range(len(envelope)):
        if envelope[i] < amp_threshold:
            # 进入低幅度区域，标记起始点
            if low_amp_start is None:
                low_amp_start = i
        else:
            # 离开低幅度区域，计算持续时间
            if low_amp_start is not None:
                low_amp_end = i
                duration = (low_amp_end - low_amp_start) / fs
                # 判定是否满足事件持续阈值
                if duration >= duration_threshold:
                    # 检查片段内是否有呼吸峰值（区分暂停/低通气）
                    peaks_in_segment = [p for p in peaks if low_amp_start <= p < low_amp_end]
                    event_type = "apnea" if len(peaks_in_segment) == 0 else "hypopnea"
                    # 记录事件信息
                    apnea_events.append({
                        "event_type": event_type,
                        "start_time": time[low_amp_start],  # 事件起始时间（s）
                        "end_time": time[low_amp_end],  # 事件结束时间（s）
                        "duration": duration,  # 事件持续时间（s）
                        "detection_method": "rule_based"  # 检测方式
                    })
                low_amp_start = None  # 重置起始点

    # 2. 基于模型的事件检测（若提供训练好的模型）
    if model is not None and scaler is not None:
        # 提取事件特征
        X_event = extract_event_features(breathing_features)
        if len(X_event) == 0:
            return apnea_events

        # 模型预测
        X_event_scaled = scaler.transform(X_event)
        y_pred = model.predict(X_event_scaled)
        y_pred_proba = model.predict_proba(X_event_scaled)[:, 1]  # 事件概率

        # 映射预测结果到时间轴
        window_size = 5  # 与特征提取窗口一致
        for i, (pred, proba) in enumerate(zip(y_pred, y_pred_proba)):
            if pred == 1:  # 预测为事件（1=事件，0=正常）
                start_time = i * window_size
                end_time = (i + 1) * window_size
                # 避免与规则检测重复
                is_duplicate = any(
                    (event["start_time"] <= start_time <= event["end_time"])
                    for event in apnea_events
                )
                if not is_duplicate:
                    apnea_events.append({
                        "event_type": "apnea/hypopnea",
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": window_size,
                        "confidence": proba,  # 预测置信度
                        "detection_method": "model_based"
                    })

    return apnea_events


if __name__ == "__main__":
    # 示例：检测10秒信号中的呼吸事件
    fs = 48000
    total_duration = 10.0

    # 1. 加载呼吸特征
    single_sweep, _ = generate_swept_sinusoid()
    transmit_signal = generate_continuous_signal(single_sweep, total_duration=total_duration, fs=fs)
    received_signal = receive_reflect_signal(total_duration=total_duration, fs=fs)
    preprocessed_signal = preprocess_signal(received_signal, transmit_signal, fs=fs)
    freq_shift, time = calculate_frequency_shift(preprocessed_signal, transmit_signal, fs=fs)
    distance_change = map_to_distance_change(freq_shift)
    breathing_features = extract_breathing_features(distance_change, time, fs=fs)

    # 2. 检测事件（基于规则）
    apnea_events = detect_apnea_events(breathing_features)

    # 3. 输出结果
    if len(apnea_events) == 0:
        print("未检测到呼吸暂停/低通气事件")
    else:
        print(f"检测到{len(apnea_events)}个呼吸事件：")
        for idx, event in enumerate(apnea_events, 1):
            print(
                f"事件{idx} | 类型：{event['event_type']} | 起始：{event['start_time']:.1f}s | 持续：{event['duration']:.1f}s")