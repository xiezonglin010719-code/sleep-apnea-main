# src/preprocessing/steps/dataset.py
import torch
from torch.utils.data import Dataset
class SignalDatasetnew(Dataset):
    def __init__(self, data, classes, mean, std):
        self.data = data
        self.classes = classes  # 标签名到索引的映射（如{"Hypopnea":0, ...}）
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # 处理图像特征
        img = torch.FloatTensor(sample["data"]).unsqueeze(0)  # (1, H, W)
        img = img / 255.0  # 归一化到[0,1]
        img = (img - self.mean) / self.std  # 标准化

        # 直接返回类别索引（CrossEntropyLoss 要求的格式）
        label = torch.LongTensor([sample["label"]])  # 单维度索引（如 tensor([2])）
        return img, label