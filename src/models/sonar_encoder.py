import os

import torch
import torch.nn as nn
import numpy as np
from scipy.signal import spectrogram
from torch.utils.data import Dataset, DataLoader
from src.models.generator_sonar import SonarFeatureGenerator
# 或
# from .generator_sonar import SonarFeatureGenerator


# 声纳数据集加载
class SonarDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        sig = data["signal"]
        label = data["label"]
        # 提取声纳特征（与PSG帧化对应，每0.5s一帧，128帧）
        frames = self._frame_signal(sig, frame_len=50, frame_hop=25)  # 100Hz→50点/0.5s
        feats = self._extract_frame_feats(frames)  # 每帧6维特征（如能量、频率等）
        return torch.FloatTensor(feats), torch.LongTensor([label])

    def _frame_signal(self, sig, frame_len, frame_hop):
        n_frames = (len(sig) - frame_len) // frame_hop + 1
        frames = np.stack([sig[i * frame_hop:i * frame_hop + frame_len] for i in range(n_frames)], axis=0)
        return frames[:128]  # 对齐PSG的128帧

    def _extract_frame_feats(self, frames):
        # 每帧提取6维特征（能量、峰峰值、重心频率等）
        feats = []
        for frame in frames:
            energy = np.sum(frame ** 2) / len(frame)
            peak_to_peak = np.max(frame) - np.min(frame)
            f, _, Sxx = spectrogram(frame, fs=100, nperseg=32)
            centroid = np.sum(f * np.mean(Sxx, axis=1)) / np.sum(np.mean(Sxx, axis=1)) if np.sum(Sxx) > 0 else 0
            feats.append([energy, peak_to_peak, centroid, np.std(frame), np.mean(frame), np.max(np.abs(frame))])
        return np.array(feats)


# 声纳编码器（映射到PSG的潜在空间z）
class SonarToZEncoder(nn.Module):
    def __init__(self, input_feat_dim=6, z_dim=256):  # z_dim需与PSG模型的_z_dim一致
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_feat_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, z_dim)
        )

    def forward(self, x):
        # x: (B, T=128, F=6)
        x = x.mean(dim=1)  # 简化：时序平均 -> (B, 6)
        return self.encoder(x)  # -> (B, z_dim)


# 训练声纳编码器（与PSG的z空间对齐）
def train_sonar_encoder(pretrained_psg_model_path, sonar_data_dir, epochs=20):
    # 加载预训练的PSG模型（获取分类头）
    psg_model = SonarFeatureGenerator()
    psg_model.load_state_dict(torch.load(pretrained_psg_model_path))
    psg_model.eval()
    z_dim = psg_model._z_dim  # 共享的潜在空间维度

    # 初始化声纳编码器
    sonar_encoder = SonarToZEncoder(z_dim=z_dim)
    dataset = SonarDataset(sonar_data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(sonar_encoder.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        sonar_encoder.train()
        total_loss = 0
        for sonar_feats, labels in dataloader:
            # 声纳特征→z
            z_sonar = sonar_encoder(sonar_feats)
            # 复用PSG模型的分类头预测
            logits = psg_model.classifier(z_sonar)
            loss = criterion(logits, labels.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

    torch.save(sonar_encoder.state_dict(), "sonar_to_z_encoder.pth")
    return sonar_encoder