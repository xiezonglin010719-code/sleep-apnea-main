import numpy as np
import os
from scipy.signal import spectrogram, butter, filtfilt
import matplotlib.pyplot as plt


class SonarSimulator:
    def __init__(self, fs=100, duration=30):
        self.fs = fs  # 声纳采样率
        self.duration = duration  # 每段30s，与PSG分段一致
        self.n_samples = fs * duration

    def _butter_bandpass(self, lowcut, highcut, order=4):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def _filter_signal(self, sig, lowcut=0.1, highcut=10):
        b, a = self._butter_bandpass(lowcut, highcut)
        return filtfilt(b, a, sig)

    def simulate_normal(self):
        """生成正常呼吸声纳信号：规律低频波动+轻微噪声"""
        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        # 基础呼吸频率（15-20次/分钟 → 0.25-0.33Hz）
        resp_freq = np.random.uniform(0.25, 0.33)
        base = np.sin(2 * np.pi * resp_freq * t) * 0.5  # 基础呼吸波形
        # 叠加轻微心跳干扰（1Hz左右）
        heart = np.sin(2 * np.pi * 1.0 * t) * 0.1
        # 环境噪声
        noise = np.random.normal(0, 0.05, self.n_samples)
        sig = self._filter_signal(base + heart + noise)
        return sig.clip(-1, 1)  # 归一化

    def simulate_osa(self):
        """生成阻塞性呼吸暂停声纳信号：暂停段+打鼾爆发"""
        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        sig = np.zeros_like(t)

        # 随机插入2-3个阻塞性事件（持续5-10s）
        for _ in range(np.random.randint(2, 4)):
            start = np.random.randint(0, self.n_samples // 2)
            duration = int(np.random.uniform(5, 10) * self.fs)
            end = min(start + duration, self.n_samples)
            # 阻塞期间信号减弱（暂停）
            sig[start:end] = np.random.normal(0, 0.02, end - start)

        # 修复：打鼾点数不超过总样本数，且支持放回采样（因为打鼾可以重叠）
        max_snore_points = min(5000, self.n_samples)  # 最多不超过总样本数
        snore_times = np.random.choice(
            self.n_samples,
            size=max_snore_points,
            replace=True  # 允许重复采样（符合实际打鼾可能连续出现的情况）
        )
        sig[snore_times] += np.random.uniform(0.8, 1.2, max_snore_points)  # 强脉冲
        sig = self._filter_signal(sig, lowcut=5, highcut=20)  # 打鼾高频特性
        return sig.clip(-1, 1)

    def generate_dataset(self, save_dir, n_normal=1000, n_osa=500):
        """生成带标签的模拟声纳数据集"""
        os.makedirs(save_dir, exist_ok=True)
        # 正常样本（标签0）
        for i in range(n_normal):
            sig = self.simulate_normal()
            np.savez_compressed(
                os.path.join(save_dir, f"normal_{i}.npz"),
                signal=sig, label=0
            )
        # OSA样本（标签2，对应配置中的Obstructive Apnea）
        for i in range(n_osa):
            sig = self.simulate_osa()
            np.savez_compressed(
                os.path.join(save_dir, f"osa_{i}.npz"),
                signal=sig, label=2
            )
        print(f"生成完成：{n_normal}正常 + {n_osa}OSA，保存至{save_dir}")


# 生成示例
if __name__ == "__main__":
    simulator = SonarSimulator()
    simulator.generate_dataset("simulated_sonar_data")
    # 可视化一个样本
    sig_normal = simulator.simulate_normal()
    sig_osa = simulator.simulate_osa()
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(sig_normal[:500])
    plt.title("Normal")
    plt.subplot(122)
    plt.plot(sig_osa[:500])
    plt.title("OSA")
    plt.savefig("sonar_samples.png")