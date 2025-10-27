# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Dict, List, Optional

class TemporalRespNet(nn.Module):
    """
    Conv1d(沿时间) + GRU + 分类头
    输入:  (B, T, F)
    输出:  {'logits': (B, C), 'z': (B, H), 'sonar': (B, D)}
    - 不在入口做时间池化，避免早期丢失事件动态
    - 'sonar' 作为可选的特征投影（若你需要生成128维声纳特征）
    """
    def __init__(
        self,
        input_dim: int,      # F = 每帧特征维度 * 模态数量
        num_classes: int = 3,
        conv_channels: List[int] = [64, 128],
        conv_kernel: int = 5,
        conv_dropout: float = 0.1,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        proj_dim: int = 128,    # sonar 投影维度
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # 线性升维: F -> C0
        c0 = conv_channels[0]
        self.in_proj = nn.Linear(input_dim, c0)

        # Conv1d 堆（沿时间维卷积：输入 (B, C, T)）
        convs = []
        in_c = c0
        for c in conv_channels:
            convs += [
                nn.Conv1d(in_c, c, kernel_size=conv_kernel, padding=conv_kernel // 2),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
                nn.Dropout(conv_dropout),
            ]
            in_c = c
        self.conv = nn.Sequential(*convs)

        # GRU（沿时间）
        self.gru = nn.GRU(
            input_size=in_c,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        z_dim = gru_hidden * 2

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # sonar 特征投影（可选）
        self.proj = nn.Sequential(
            nn.Linear(z_dim, proj_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B, T, F)
        """
        assert x.dim() == 3, f"Expect (B,T,F), got {x.shape}"
        B, T, F = x.shape

        # 先线性到 C0，再转为 (B, C0, T) 供 Conv1d
        h = self.in_proj(x)             # (B, T, C0)
        h = h.transpose(1, 2)           # (B, C0, T)
        h = self.conv(h)                # (B, Ck, T)
        h = h.transpose(1, 2)           # (B, T, Ck)

        # GRU
        z, _ = self.gru(h)              # (B, T, 2*H)
        z = z.mean(dim=1)               # 时间平均 (B, 2H)

        logits = self.classifier(z)     # (B, C)
        sonar = self.proj(z)            # (B, D)
        return {"logits": logits, "z": z, "sonar": sonar}
