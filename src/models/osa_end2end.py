# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):  # (B, T, H)
        scores = self.v(torch.tanh(self.W(lstm_output)))   # (B, T, 1)
        attn = F.softmax(scores, dim=1)                    # (B, T, 1)
        ctx = torch.sum(lstm_output * attn, dim=1)         # (B, H)
        return ctx, attn


def _norm2d(c, use_groupnorm=True, gn_groups=8):
    if use_groupnorm:
        return nn.GroupNorm(num_groups=min(gn_groups, c), num_channels=c, affine=True)
    else:
        return nn.BatchNorm2d(c)


class OSAEnd2EndModel(nn.Module):
    """
    CNN(2D) → LSTM → Attention → MLP
    - GroupNorm（默认开）更稳
    - CNN 对 (B,T,1,H,W) 并行
    - 提供 set_output_prior_bias（用于二分类先验）
    """
    def __init__(self, img_channels=1, img_size=64, seq_len=10, num_classes=3,
                 use_groupnorm=True, gn_groups=8,
                 lstm_hidden=128, lstm_layers=2, lstm_dropout=0.3,
                 mlp_hidden=64, mlp_drop=0.3):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes

        # ---- CNN backbone ----
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, 1, 1),
            _norm2d(32, use_groupnorm), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, 1, 1),
            _norm2d(64, use_groupnorm), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            _norm2d(128, use_groupnorm), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            _norm2d(256, use_groupnorm), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )

        # ---- 推一次，确定展平维度 ----
        self.cnn_feature_dim = self._infer_cnn_dim(img_channels, img_size)

        # ---- LSTM + Attention ----
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=(lstm_dropout if lstm_layers > 1 else 0.0)
        )
        self.attention = AttentionBlock(lstm_hidden)

        # ---- 分类头 ----
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_drop),
            nn.Linear(mlp_hidden, num_classes)
        )

        self._init_weights()

    def _infer_cnn_dim(self, img_channels, img_size):
        was_training = self.cnn.training
        try:
            self.cnn.eval()
            with torch.no_grad():
                x = torch.zeros(1, img_channels, img_size, img_size)
                y = self.cnn(x)
                return int(y.numel())
        finally:
            if was_training:
                self.cnn.train()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def set_output_prior_bias(self, count_neg, count_pos):
        """
        仅二分类时可用：按先验设置最终层 bias。三分类下不使用。
        """
        if self.num_classes != 2:
            return
        p1 = count_pos / max(1, (count_neg + count_pos))
        prior = math.log(max(p1, 1e-6) / max(1 - p1, 1e-6))
        last = None
        for m in reversed(list(self.classifier.modules())):
            if isinstance(m, nn.Linear) and m.out_features == 2:
                last = m
                break
        if last is not None:
            if last.bias is None:
                last.bias = nn.Parameter(torch.zeros(2, device=last.weight.device))
            last.bias.data[1] = prior / 2
            last.bias.data[0] = -prior / 2
            print(f"[Init] Set final bias prior≈{prior:.4f} using counts neg/pos={count_neg}/{count_pos}")

    def forward(self, x):
        """
        x: (B, T, C=1, H, W)
        返回未归一化 logits: (B, num_classes)
        """
        B, T, C, H, W = x.shape
        x_ = x.view(B * T, C, H, W)        # (B*T, C, H, W)
        f_ = self.cnn(x_)                  # (B*T, 256, h, w)
        f_ = f_.view(B * T, -1)            # (B*T, F)
        f = f_.view(B, T, -1)              # (B, T, F)

        lstm_out, _ = self.lstm(f)         # (B, T, H)
        ctx, _ = self.attention(lstm_out)  # (B, H)
        out = self.classifier(ctx)         # (B, num_classes)
        return out
