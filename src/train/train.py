# -*- coding: utf-8 -*-
import os
import argparse
import pickle
from pathlib import Path
from typing import Any, List, Optional, Dict, Tuple, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.train.train_eval import train_epoch, eval_model
from src.models.baseline_cnn import CNN
from src.models.pretrained_convnext import CustomConvNeXt
from src.preprocessing.steps.dataset import SignalDataset
from src.preprocessing.steps.config import load_config


# -------------------------
# 工具：安全配置读取 + 路径解析（Py3.8 兼容）
# -------------------------
def cfg_get(cfg: Any, path: str, default=None):
    """
    兼容 Pydantic/对象/字典 的安全取值：path 形如 'training.batch_size'
    """
    cur = cfg
    for key in path.split('.'):
        if cur is None:
            return default
        if hasattr(cur, key):
            cur = getattr(cur, key)
            continue
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
            continue
        return default
    return cur

def project_root() -> Path:
    # 假设本文件位于 .../src/train/train.py
    return Path(__file__).resolve().parents[2]

def resolve_dir(p: Optional[str], anchors: List[Path]) -> str:
    """
    把可能的相对路径 p 转为绝对路径。
    anchors: 依次尝试的锚点目录（如 CWD、项目根等）
    """
    if p is None:
        return ""
    P = Path(p)
    if P.is_absolute():
        return str(P)
    for a in anchors:
        cand = a / P
        if cand.exists():
            return str(cand.resolve())
    return str((anchors[0] / P).resolve())


# -------------------------
# 读取 .pickle & 标签映射
# -------------------------
def _iter_pickle_events(data_dir: str) -> Iterable:
    """遍历目录下所有 .pickle，逐个 yield 事件对象"""
    data_dir = str(Path(data_dir).resolve())
    for filename in os.listdir(data_dir):
        if filename.endswith(".pickle"):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'rb') as f:
                events = pickle.load(f)
                for ev in events:
                    yield ev

def build_label_map(*dirs: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    扫描多个目录的 .pickle，收集所有出现过的标签（字符串或数字），
    返回 (label_to_index, index_to_label)
    """
    uniq = []
    seen = set()
    for d in dirs:
        if not d:
            continue
        p = Path(d)
        if not p.exists():
            continue
        for ev in _iter_pickle_events(str(p)):
            key = str(ev.label)  # 统一成字符串用于去重
            if key not in seen:
                seen.add(key)
                uniq.append(key)

    if len(uniq) == 0:
        raise ValueError("未在任一目录中发现标签，检查 .pickle 内容")

    # 稳定排序，避免不同环境下顺序不一致
    uniq = sorted(uniq)
    label_to_index = {name: i for i, name in enumerate(uniq)}
    index_to_label = {i: name for name, i in label_to_index.items()}
    print("标签映射（index -> name）:", index_to_label)
    return label_to_index, index_to_label

def load_pickle_data(data_dir: str, label_to_index: Dict[str, int]):
    """
    加载 .pickle -> [{"data": uint8_img, "label": int_index}, ...], classes: index->name
    """
    dataset = []
    classes = {idx: name for name, idx in label_to_index.items()}  # index -> name
    data_dir = str(Path(data_dir).resolve())

    def to_uint8_image(arr):
        arr = np.asarray(arr)
        # 若是 (C,H,W) 且 C 为 1 或 3，转为 (H,W,C)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        # 归一化到 [0,255] 并转 uint8（支持 float/其他范围）
        if arr.dtype != np.uint8:
            a = arr.astype(np.float32)
            a_min, a_max = float(np.min(a)), float(np.max(a))
            if a_max > a_min:
                a = (a - a_min) / (a_max - a_min)
            else:
                a = np.zeros_like(a)
            arr = (a * 255.0).clip(0, 255).astype(np.uint8)
        return arr

    for filename in os.listdir(data_dir):
        if filename.endswith(".pickle"):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'rb') as f:
                events = pickle.load(f)
                for ev in events:
                    img = to_uint8_image(ev.signal)
                    # 将事件的标签转换为字符串，再查映射表
                    key = str(ev.label)  # 确保是字符串格式（如 'normal'、'apnea'、'2' 等）
                    if key not in label_to_index:
                        raise KeyError(f"未登记的标签: {ev.label}（文件: {filename}），请检查数据或更新映射")
                    lab_idx = label_to_index[key]  # 转换为整数索引
                    dataset.append({"data": img, "label": lab_idx})  # 存储整数标签
    return dataset, classes


# -------------------------
# collate_fn 工厂：规范形状 + 标签越界检查
# 返回 (X: float32 [B,C,H,W], Y: long [B]) 适配 CrossEntropyLoss
# -------------------------
def make_collate_fn(num_classes: int, name2idx: Optional[Dict[str, int]] = None):
    def _to_chw_tensor(x: torch.Tensor) -> torch.Tensor:
        """
        规范为 [C,H,W]；baseline CNN 期望单通道：
        - 2D -> [1,H,W]
        - [H,W,C]/[H,C,W]/[C,H,W] 自动转 [C,H,W]
        - 若 3 通道 -> 压成 1 通道（均值）；若其他通道数 -> 取前 1 通道
        """
        if x.ndim == 2:
            return x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"Unexpected image ndim={x.ndim}, shape={tuple(x.shape)}")

        if x.shape[0] in (1, 3) and x.shape[1] > 8 and x.shape[2] > 8:
            chw = x
        elif x.shape[2] in (1, 3) and x.shape[0] > 8 and x.shape[1] > 8:
            chw = x.permute(2, 0, 1)
        elif x.shape[1] in (1, 3) and x.shape[0] > 8 and x.shape[2] > 8:
            chw = x.permute(1, 0, 2)
        else:
            chw = x.permute(2, 0, 1)

        if chw.shape[0] == 3:
            chw = chw.mean(dim=0, keepdim=True)
        elif chw.shape[0] != 1:
            chw = chw[:1, ...]
        return chw

    def _collate(batch):
        xs, ys = [], []

        for item in batch:
            # 兼容 dict 或 (x,y)
            if isinstance(item, dict):
                x = item.get("data")
                y = item.get("label")
            else:
                x, y = item

            # ---- X：转 tensor & 规范到 [C,H,W] ----
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.tensor(x)
            x = x.float()
            x = _to_chw_tensor(x)

            # ---- Y：严格转换为整数索引 ----
            if isinstance(y, (np.ndarray, torch.Tensor)):
                y = int(y.item())
            elif isinstance(y, str):
                # 先处理纯数字字符串
                if y.isdigit() or (y.startswith('-') and y[1:].isdigit()):
                    y = int(y)
                else:
                    # 必须在 name2idx 中
                    if name2idx is None:
                        raise ValueError(f"检测到字符串标签 '{y}'，但未提供 name2idx 映射。")
                    if y not in name2idx:
                        raise KeyError(f"未登记的字符串标签 '{y}'，请检查数据或更新映射表。")
                    y = name2idx[y]
            else:
                y = int(y)

            if not (0 <= y < num_classes):
                raise IndexError(f"Label {y} 超出范围 0..{num_classes-1}，请检查数据与映射是否一致。")

            xs.append(x)
            ys.append(y)

        X = torch.stack(xs, dim=0)                # [B, C, H, W]
        Y = torch.tensor(ys, dtype=torch.long)    # [B]
        return X, Y

    return _collate


# -------------------------
# 主流程
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../../config.yaml")
    parser.add_argument("--model_type", choices=["baseline", "convnext"], default="baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_path", default="models/")
    parser.add_argument("--train_pickle_dir", default=None)
    parser.add_argument("--val_pickle_dir", default=None)
    # 旧的 --num_classes 已废弃，完全以数据决定
    args = parser.parse_args()

    # 解析配置，且让相对路径在 CWD/项目根中解析
    config_path = resolve_dir(args.config, [Path.cwd(), project_root()])
    config = load_config(config_path)

    signals_path = cfg_get(config, 'paths.signals_path', 'data/processed/signals')
    anchors = [Path.cwd(), project_root()]
    train_data_dir = resolve_dir(args.train_pickle_dir or signals_path, anchors)
    val_data_dir   = resolve_dir(args.val_pickle_dir   or signals_path, anchors)

    if not Path(train_data_dir).exists():
        raise FileNotFoundError(f"train_pickle_dir 不存在：{train_data_dir}")
    if not Path(val_data_dir).exists():
        raise FileNotFoundError(f"val_pickle_dir 不存在：{val_data_dir}")

    # 统一标签映射（扫描 训练+验证 两个目录）
    label_to_index, index_to_label = build_label_map(train_data_dir, val_data_dir)
    num_classes = len(label_to_index)
    print(f"[INFO] 数据包含 {num_classes} 个类别：{index_to_label}")

    # 读取数据（此时样本中的 label 已经是 int 索引）
    print(f"加载训练集数据 from {train_data_dir}...")
    train_dataset_list, train_classes = load_pickle_data(train_data_dir, label_to_index)
    print(f"加载验证集数据 from {val_data_dir}...")
    val_dataset_list,   val_classes   = load_pickle_data(val_data_dir,   label_to_index)

    print(f"训练集样本数: {len(train_dataset_list)}")
    print(f"验证集样本数: {len(val_dataset_list)}")
    if len(train_dataset_list) == 0 or len(val_dataset_list) == 0:
        raise ValueError("未加载到数据，请检查 .pickle 文件路径或文件内容")

    # 构造 SignalDataset（内部会访问 sample['data']；我们已按此键名组织）
    train_dataset = SignalDataset(
        dataset=train_dataset_list,
        classes=train_classes,
        mean=None,
        std=None
    )
    val_dataset = SignalDataset(
        dataset=val_dataset_list,
        classes=val_classes,
        mean=None,
        std=None
    )

    # 训练配置（带默认值，避免 AttributeError）
    input_size   = cfg_get(config, 'audio.image_size', (1, 224, 224))
    batch_size   = cfg_get(config, 'training.batch_size', 16)
    num_workers  = cfg_get(config, 'training.num_workers', 0)
    lr           = cfg_get(config, 'training.learning_rate', 1e-3)
    weight_decay = cfg_get(config, 'training.weight_decay', 0.0)

    # DataLoader（绑定严格的 collate_fn）
    collate_fn = make_collate_fn(num_classes, name2idx=label_to_index)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    # 设备 & 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if args.model_type == "baseline":
        model = CNN(
            num_classes=num_classes,
            input_size=input_size
        ).to(device)
    else:
        model = CustomConvNeXt(
            num_classes=num_classes
        ).to(device)

    # ✅ 多分类固定用 CrossEntropyLoss（模型输出 [B,num_classes]；标签 long [B]）
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # 训练循环
    os.makedirs(args.save_path, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1     = eval_model(model, val_loader, criterion, device, cm=False)

        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_path = os.path.join(args.save_path, f"{args.model_type}_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path} (F1: {best_f1:.4f})")

    final_path = os.path.join(args.save_path, f"{args.model_type}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
