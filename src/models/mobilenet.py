import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        # 深度可分离卷积
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        # 投影层
        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
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

        # 现在可安全访问通道数
        in_channels = input_size[0]
        # 初始卷积层
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            # 倒残差块组
            InvertedResidual(32, 16, 1, 1),
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6),
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
        )
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x