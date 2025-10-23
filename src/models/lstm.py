import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    def __init__(self, num_classes, input_size=(1, 224, 224)):
        super().__init__()

        # 新增：处理 input_size 格式，确保为 (C, H, W) 三元组
        if isinstance(input_size, int):
            # 若输入是整数（如 224），默认单通道，转为 (1, 224, 224)
            input_size = (1, input_size, input_size)
        elif isinstance(input_size, (list, tuple)):
            if len(input_size) == 2:
                # 若输入是 (224, 224)，补全通道数为 1
                input_size = (1, input_size[0], input_size[1])
            elif len(input_size) != 3:
                raise ValueError(f"input_size 必须是 3 元素元组 (C, H, W)，当前为 {input_size}")

        # 现在可以安全地通过下标访问
        in_channels = input_size[0]
        self.input_h, self.input_w = input_size[1], input_size[2]

        # CNN特征提取部分
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # 计算CNN输出维度（修正：确保 input_h 和 input_w 是整数且可被8整除）
        self.cnn_output_dim = 128 * (self.input_h // 8) * (self.input_w // 8)
        self.lstm_input_dim = 256  # LSTM输入特征数
        self.lstm_hidden_dim = 128  # LSTM隐藏层大小
        self.num_lstm_layers = 2  # LSTM层数

        # 维度适配层：将CNN输出映射到LSTM输入维度
        self.fc_adapt = nn.Linear(self.cnn_output_dim, self.lstm_input_dim)

        # LSTM层（双向）
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim * 2, 64),  # 双向LSTM输出×2
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # CNN特征提取：[B, C, H, W] → [B, 128, H//8, W//8]
        cnn_feat = self.cnn(x)
        # 展平为时序特征：[B, 128*(H//8)*(W//8)] → [B, 1, lstm_input_dim]
        seq_feat = self.fc_adapt(cnn_feat.flatten(1)).unsqueeze(1)
        # LSTM处理：[B, 1, lstm_input_dim] → [B, 1, 2*lstm_hidden_dim]
        lstm_out, _ = self.lstm(seq_feat)
        # 分类：取最后一个时序步的输出
        out = self.classifier(lstm_out[:, -1, :])
        return out