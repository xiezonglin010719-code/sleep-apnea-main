import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """注意力模块：聚焦关键时序特征"""
    def __init__(self, hidden_dim):
        super(AttentionBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_dim)
        scores = self.v(torch.tanh(self.W(lstm_output)))  # (batch_size, seq_len, 1)
        attn_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        weighted_output = torch.sum(lstm_output * attn_weights, dim=1)  # (batch_size, hidden_dim)
        return weighted_output, attn_weights

class OSAEnd2EndModel(nn.Module):
    """端到端OSA诊断模型：CNN提取空间特征 + LSTM提取时序特征 + Attention强化关键信息"""
    def __init__(self, img_channels=1, img_size=224, seq_len=10, num_classes=2):
        super(OSAEnd2EndModel, self).__init__()
        self.seq_len = seq_len

        # CNN模块：提取梅尔频谱的空间特征
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # 计算CNN输出特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, img_channels, img_size, img_size)
            cnn_out = self.cnn(dummy_input)
            self.cnn_feature_dim = cnn_out.view(1, -1).size(1)

        # LSTM模块：提取时序特征
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        # 注意力模块
        self.attention = AttentionBlock(128)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, 1, img_size, img_size) -> 输入为时序梅尔频谱序列
        batch_size = x.size(0)

        # 1. CNN提取每个时间步的空间特征
        cnn_features = []
        for t in range(self.seq_len):
            frame = x[:, t, :, :, :]  # (batch_size, 1, img_size, img_size)
            feat = self.cnn(frame)  # (batch_size, 256, H, W)
            feat = feat.view(batch_size, -1)  # (batch_size, cnn_feature_dim)
            cnn_features.append(feat)
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch_size, seq_len, cnn_feature_dim)

        # 2. LSTM提取时序特征
        lstm_out, _ = self.lstm(cnn_features)  # (batch_size, seq_len, 128)

        # 3. 注意力强化关键特征
        attn_out, _ = self.attention(lstm_out)  # (batch_size, 128)

        # 4. 分类输出
        output = self.classifier(attn_out)  # (batch_size, num_classes)
        return output