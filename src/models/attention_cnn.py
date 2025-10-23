import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """通道注意力+空间注意力融合模块"""

    def __init__(self, channels):
        super().__init__()
        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        # 通道注意力
        ca = self.channel_attn(x)
        x = x * ca
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        return x + residual


class AttentionResCNN(nn.Module):
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
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 注意力残差块组
        self.layer1 = nn.Sequential(
            AttentionBlock(32),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            AttentionBlock(64),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            AttentionBlock(128),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x).flatten(1)
        x = self.classifier(x)
        return x