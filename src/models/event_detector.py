import torch
import torch.nn as nn
import torch.nn.functional as F


class EventDetector(nn.Module):
    """第一阶段：事件检测模型（鼾声/低通气/正常呼吸/噪声）"""

    def __init__(self, num_classes=4, input_size=(1, 224, 224)):
        super().__init__()
        # 处理输入尺寸
        if isinstance(input_size, int):
            input_size = (1, input_size, input_size)
        elif len(input_size) == 2:
            input_size = (1, input_size[0], input_size[1])
        self.in_channels = input_size[0]

        # 特征提取 backbone
        self.features = nn.Sequential(
            # 增强低频特征捕捉（适应呼吸音特性）
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 深度可分离卷积块
            self._make_depthwise_block(64, 128, stride=2),
            self._make_depthwise_block(128, 128, stride=1),
            self._make_depthwise_block(128, 256, stride=2),
            self._make_depthwise_block(256, 256, stride=1),
            self._make_depthwise_block(256, 512, stride=2),

            # 多尺度特征融合
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True)
        )

        # 事件分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def _make_depthwise_block(self, in_channels, out_channels, stride):
        """深度可分离卷积块"""
        return nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # 逐点卷积
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x