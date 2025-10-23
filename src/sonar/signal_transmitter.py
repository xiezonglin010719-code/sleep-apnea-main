#步骤2
import sounddevice as sd
import numpy as np
from typing import Optional
from signal_generator import generate_swept_sinusoid, generate_continuous_signal


def get_speaker_info() -> dict:
    """
    获取扬声器设备信息，适配不同手机硬件：
    - 自动识别默认输出设备（智能手机扬声器）
    - 提取设备参数（通道数、默认采样率），确保信号兼容性
    """
    devices = sd.query_devices()
    output_device_id = sd.default.device[1]  # 默认输出设备（扬声器）
    device_info = devices[output_device_id]
    return {
        "device_id": output_device_id,
        "max_channels": device_info["max_output_channels"],
        "default_fs": device_info["default_samplerate"],
        "device_name": device_info["name"]
    }


def transmit_signal(
        signal: np.ndarray,
        fs: int = 48000,
        device_id: Optional[int] = None,
        volume: float = 0.8  # 输出音量（避免硬件过载，参考论文校准逻辑）
) -> None:
    """
    发射扫频信号，参考论文：
    1. 复用手机内置扬声器，无需改装硬件
    2. 信号功率通过音量参数自适应调整，避免不同手机硬件差异导致的失真
    3. 支持实时发射状态监控，确保信号完整输出
    """
    # 音量校准（限制信号在[-1,1]范围内，防止扬声器过载）
    signal_calibrated = signal * volume
    signal_calibrated = np.clip(signal_calibrated, -1.0, 1.0)

    try:
        # 发射信号（阻塞模式，等待发射完成）
        sd.play(signal_calibrated, samplerate=fs, device=device_id)
        print(f"信号发射中 | 设备：{device_id} | 采样率：{fs}Hz | 音量：{volume}")
        sd.wait()  # 等待信号发射完成
        print("信号发射完成（无硬件改装，符合论文设计）")
    except Exception as e:
        raise RuntimeError(f"扬声器发射失败：{str(e)}（请检查手机扬声器权限）") from e


if __name__ == "__main__":
    # 示例：生成并发射10秒扫频信号
    # 1. 生成信号
    single_sweep, fs = generate_swept_sinusoid()
    continuous_sweep = generate_continuous_signal(single_sweep, total_duration=10.0, fs=fs)

    # 2. 获取扬声器信息
    speaker_info = get_speaker_info()
    print(f"使用扬声器 | 名称：{speaker_info['device_name']} | ID：{speaker_info['device_id']}")

    # 3. 发射信号
    transmit_signal(continuous_sweep, fs=fs, device_id=speaker_info["device_id"])