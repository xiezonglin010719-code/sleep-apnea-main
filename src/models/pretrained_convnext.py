import torch
import torch.nn as nn
import torchvision.models as models


class CustomConvNeXt(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNeXt, self).__init__()
        self.convnext = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        # 替换最后一层以适配你的类别数
        self.convnext.classifier[2] = nn.Linear(self.convnext.classifier[2].in_features, num_classes)

    def forward(self, x):
        return self.convnext(x)
