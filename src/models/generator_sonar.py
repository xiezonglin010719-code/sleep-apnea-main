import torch
import torch.nn as nn
from typing import List, Dict, Optional

import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):
    """自适应通道的时序编码：LazyConv1d + BiLSTM"""
    def __init__(self, conv_channels: int = 64, lstm_hidden: int = 128, lstm_layers: int = 1):
        super().__init__()
        # 第一层用 LazyConv1d：不需要预先知道 in_channels（F）
        self.conv = nn.Sequential(
            nn.LazyConv1d(out_channels=conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.out_dim = lstm_hidden * 2

    def forward(self, x_btF: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x_btF.transpose(1, 2)   # (B, F, T)
        x = self.conv(x)            # (B, C, T)
        x = x.transpose(1, 2)       # (B, T, C)
        out, _ = self.lstm(x)       # (B, T, 2H)
        return out


class SonarFeatureGenerator(nn.Module):
    """
    改进版：带时间卷积 + BiLSTM + Attention 池化的分类/生成器
    - 输入：(B, T, F) 或 (B, F)
    - 分类输出：logits (B, num_classes)
    - 生成输出：sonar (B, output_dim)
    """
    def __init__(
        self,
        input_dim: int = 36,              # F
        output_dim: int = 128,
        hidden_layers: List[int] = [256], # MLP 的隐藏层（在时序编码之后）
        dropout_rate: float = 0.3,
        activation: str = "relu",
        num_classes: int = 3,
        temporal_pool: str = "attention",  # 默认 attention
        conv_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.temporal_pool = temporal_pool.lower()
        self.dropout_rate = dropout_rate

        # 激活函数
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.Tanh()

        # ——(A) 时序编码器：Conv1d + BiLSTM——
        # 注意：只有当输入是 (B,T,F) 时才用；若是 (B,F) 则直接走 (B,F)
        self.temporal_encoder = TemporalEncoder(
            conv_channels=conv_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers
        )
        temporal_feat_dim = self.temporal_encoder.out_dim  # 2*H

        # ——(B) Attention 池化（修复：放到 __init__，成为可训练参数）——
        self.attn_fc = nn.Linear(temporal_feat_dim, 1, bias=False)

        # ——(C) 时序后 MLP 编码器（把 pooled 向量再加工）——
        enc_layers = []
        prev_dim = temporal_feat_dim
        for h in hidden_layers:
            enc_layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), self.act, nn.Dropout(self.dropout_rate)]
            prev_dim = h
        if not enc_layers:
            # 保底：给一个线性层，不然 decoder/classifier 输入维度等于 temporal_feat_dim
            enc_layers = [nn.Identity()]
        self.encoder = nn.Sequential(*enc_layers)
        self._z_dim = prev_dim  # 编码后维度

        # ——(D) 解码器（生成 sonar 向量）——
        self.decoder = nn.Sequential(
            nn.Linear(self._z_dim, max(self._z_dim // 2, 64)),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(max(self._z_dim // 2, 64), self.output_dim),
            nn.Tanh()
        )

        # ——(E) 分类头——
        self.classifier = nn.Sequential(
            nn.Linear(self._z_dim, 64),
            self.act,
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, self.num_classes)
        )

        # ——(F)（可选）输入维度适配器：如果你有可能传 (B,F')——
        # 这里不用动态创建（那样会反复重置权重），用 nn.Identity 占位；
        # 如果确实会传入 F'!=input_dim 的二维输入，可改用 nn.LazyLinear 再接一个非线性。
        self.input_adapter: Optional[nn.Module] = None

    def _temporal_pool(self, H: torch.Tensor) -> torch.Tensor:
        """
        对时序编码器输出 H: (B, T, D) 做池化 -> (B, D)
        支持 mean/max/attention（默认）
        """
        if self.temporal_pool == "max":
            return H.max(dim=1).values
        elif self.temporal_pool == "mean":
            return H.mean(dim=1)
        else:
            # attention: a_t = softmax(W h_t)
            a = self.attn_fc(H).squeeze(-1)          # (B, T)
            a = torch.softmax(a, dim=1).unsqueeze(-1)  # (B, T, 1)
            return (H * a).sum(dim=1)                # (B, D)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B, T, F) 或 (B, F)
        返回: {"sonar": (B,output_dim), "logits": (B,num_classes), "z": (B,_z_dim)}
        """
        if x.dim() == 3:
            # 时序 -> 编码 -> 池化
            H = self.temporal_encoder(x)     # (B, T, D)
            g = self._temporal_pool(H)       # (B, D)
        elif x.dim() == 2:
            # 非时序，直接走适配 + 线性编码
            g = x
            if self.input_adapter is not None:
                g = self.input_adapter(g)
        else:
            g = x.view(x.size(0), -1)

        z = self.encoder(g)                  # (B, _z_dim)
        sonar = self.decoder(z)              # (B, output_dim)
        logits = self.classifier(z)          # (B, num_classes)
        return {"sonar": sonar, "logits": logits, "z": z}
