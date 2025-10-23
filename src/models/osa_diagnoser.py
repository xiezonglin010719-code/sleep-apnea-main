# src/models/osa_diagnoser.py
import torch
import torch.nn as nn


class OSADiagnoser(nn.Module):
    def __init__(self, num_diagnosis_classes=2, event_feature_dim=3):  # 新增参数：事件特征维度（默认为3）
        super(OSADiagnoser, self).__init__()
        # 事件序列处理（LSTM）
        self.lstm = nn.LSTM(
            input_size=event_feature_dim,  # 关键修正：使用事件特征维度（3）作为输入大小
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        # 统计特征处理
        self.stats_fc = nn.Sequential(
            nn.Linear(10, 32),  # 10是统计特征维度（保持不变）
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # 融合特征并输出诊断结果
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 64),  # 64（LSTM输出） + 32（统计特征）
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_diagnosis_classes)
        )

    def forward(self, event_sequence, stats_features):
        # event_sequence shape: (batch_size, window_size, event_feature_dim)
        # stats_features shape: (batch_size, 10)

        # LSTM处理事件序列
        lstm_out, _ = self.lstm(event_sequence)  # 输出 shape: (batch_size, window_size, 64)
        lstm_last = lstm_out[:, -1, :]  # 取最后一个时间步的输出

        # 处理统计特征
        stats_out = self.stats_fc(stats_features)  # 输出 shape: (batch_size, 32)

        # 融合特征
        fused = torch.cat([lstm_last, stats_out], dim=1)  # 融合后 shape: (batch_size, 64+32)
        output = self.fusion(fused)  # 输出 shape: (batch_size, num_diagnosis_classes)
        return output