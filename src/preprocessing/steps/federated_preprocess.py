# src/preprocessing/federated_preprocess.py
import os
import numpy as np
import pickle
from pathlib import Path
from scipy.signal import resample
from typing import Dict, List, Tuple
import librosa

# 配置参数
SAMPLE_RATE = 48000  # 声纳信号采样率
PSG_SAMPLE_RATE = 100  # PSG生理信号采样率
FEATURE_DIM = 128  # 特征维度
CHUNK_DURATION = 5  # 5秒切片


class FederatedDataProcessor:
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.psg_dir = self.data_root / "psg"
        self.sonar_dir = self.data_root / "sonar"
        self.processed_dir = self.data_root / "processed"
        self.processed_dir.mkdir(exist_ok=True)

    def load_psg_data(self, subject_id: str) -> Tuple[np.ndarray, List[str]]:
        """加载PSG数据并提取呼吸暂停标签"""
        psg_path = self.psg_dir / f"{subject_id}.pkl"
        with open(psg_path, "rb") as f:
            psg_data = pickle.load(f)

        # 提取标签（0:正常, 1:呼吸暂停）
        labels = psg_data["labels"]
        signals = psg_data["signals"]  # 多通道生理信号

        # 标准化长度（与声纳数据对齐）
        target_length = int(CHUNK_DURATION * SAMPLE_RATE)
        resampled_signals = resample(signals, target_length, axis=1)
        return resampled_signals, labels

    def process_sonar_signal(self, sonar_path: Path) -> np.ndarray:
        """处理原始声纳信号为梅尔频谱特征"""
        signal, _ = librosa.load(sonar_path, sr=SAMPLE_RATE)

        # 切片为5秒片段
        chunk_length = CHUNK_DURATION * SAMPLE_RATE
        chunks = [signal[i:i + chunk_length] for i in range(0, len(signal), chunk_length)]

        # 生成梅尔频谱特征
        features = []
        for chunk in chunks:
            if len(chunk) < chunk_length:
                chunk = np.pad(chunk, (0, chunk_length - len(chunk)), mode="constant")
            mel = librosa.feature.melspectrogram(
                y=chunk, sr=SAMPLE_RATE, n_mels=FEATURE_DIM
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            features.append(mel_db[np.newaxis, ...])  # 增加通道维度

        return np.concatenate(features, axis=0)  # 形状: [num_chunks, 1, FEATURE_DIM, time_steps]

    def create_federated_dataset(self, client_id: str) -> Dict:
        """为单个客户端创建联邦数据集（声纳特征+PSG标签）"""
        client_dir = self.sonar_dir / client_id
        if not client_dir.exists():
            raise FileNotFoundError(f"客户端数据不存在: {client_dir}")

        # 聚合客户端所有声纳文件
        all_features = []
        all_labels = []
        for sonar_file in client_dir.glob("*.wav"):
            subject_id = sonar_file.stem.split("_")[0]
            # 处理声纳特征
            features = self.process_sonar_signal(sonar_file)
            # 加载对应PSG标签
            _, labels = self.load_psg_data(subject_id)
            # 确保特征与标签长度匹配
            min_len = min(len(features), len(labels))
            all_features.append(features[:min_len])
            all_labels.append(labels[:min_len])

        # 合并为客户端数据集
        federated_data = {
            "features": np.concatenate(all_features, axis=0),
            "labels": np.concatenate(all_labels, axis=0),
            "client_id": client_id
        }

        # 保存到本地
        save_path = self.processed_dir / f"client_{client_id}_data.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(federated_data, f)
        return federated_data


if __name__ == "__main__":
    processor = FederatedDataProcessor(data_root="../data")
    # 示例：处理客户端1的数据
    processor.create_federated_dataset(client_id="client_001")