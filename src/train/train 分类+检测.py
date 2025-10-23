import os
import argparse
import pickle
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# 导入模型和工具函数
from src.models.event_detector import EventDetector
from src.models.osa_diagnoser import OSADiagnoser

from src.preprocessing.steps.config import load_config
from src.preprocessing.steps.datasetnew import SignalDatasetnew
from src.train.train_eval import train_epoch, eval_model
from src.utils.utils import load_pickle_events, to_uint8_image

import torch.nn.functional as F


# 工具函数
def cfg_get(cfg, path, default=None):
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
    return Path(__file__).resolve().parents[2]


def resolve_dir(p, anchors):
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


def build_label_map(*dirs):
    """构建标签映射（独立实现，不依赖外部文件）"""

    def _iter_pickle_events(data_dir: str):
        data_dir = str(Path(data_dir).resolve())
        for filename in os.listdir(data_dir):
            if filename.endswith(".pickle"):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, 'rb') as f:
                    events = pickle.load(f)
                    for ev in events:
                        yield ev

    uniq = []
    seen = set()
    for d in dirs:
        if not d:
            continue
        p = Path(d)
        if not p.exists():
            continue
        for ev in _iter_pickle_events(str(p)):
            key = str(ev.label).strip()
            if key not in seen:
                seen.add(key)
                uniq.append(key)
    if not uniq:
        raise ValueError("未发现标签")
    uniq = sorted(uniq)
    return {n: i for i, n in enumerate(uniq)}, {i: n for i, n in enumerate(uniq)}


def load_event_data(data_dir, label_map):
    dataset = []
    valid_labels = set(label_map.keys())  # 只保留有真值的标签名
    for filename in os.listdir(data_dir):
        if filename.endswith(".pickle"):
            pickle_path = os.path.join(data_dir, filename)
            events = load_pickle_events(pickle_path)
            for ev in events:
                event_label = str(ev.label).strip()
                # 只保留在真值标签中的事件
                if event_label not in valid_labels:
                    # 可选：打印无真值的标签，确认是否需要补充标注
                    # print(f"无真值标签 '{event_label}'（文件：{filename}），跳过")
                    continue

                # 处理特征和时间戳（与之前一致）
                img = to_uint8_image(ev.signal)
                label = label_map[event_label]
                event_start_time = ev.start
                subject_id = filename.split('.')[0]

                dataset.append({
                    "data": img,
                    "label": label,
                    "timestamp": event_start_time,
                    "subject_id": subject_id,
                    "end_time": ev.end
                })
    print(f"加载完成：共保留 {len(dataset)} 个有效事件（仅包含有真值的标签）")
    return dataset


def calculate_event_stats(events_window):
    """计算事件窗口的统计特征"""
    # 事件类型占比
    event_types = [np.argmax(e["probs"]) for e in events_window]
    type_counts = np.bincount(event_types, minlength=4) / len(event_types)

    # 时间特征
    start_times = [e["timestamp"] for e in events_window]
    end_times = [e["end_time"] for e in events_window]

    # 持续时间统计
    durations = [end - start for start, end in zip(start_times, end_times)]
    avg_duration = np.mean(durations) if durations else 0.0
    std_duration = np.std(durations) if durations else 0.0

    # 间隔时间统计
    intervals = []
    for i in range(1, len(start_times)):
        interval = start_times[i] - end_times[i - 1]
        intervals.append(interval)
    avg_interval = np.mean(intervals) if intervals else 0.0
    std_interval = np.std(intervals) if intervals else 0.0

    # 合并为10维特征
    return np.concatenate([
        type_counts,
        [avg_duration, std_duration],
        [avg_interval, std_interval],
        [len(events_window)],
        [sum(durations) if durations else 0.0]
    ])


def create_event_sequences(event_results, subject_id, window_size=10):
    """生成事件滑动窗口序列"""
    subject_events = sorted(
        [e for e in event_results if e["subject_id"] == subject_id],
        key=lambda x: x["timestamp"]
    )

    if len(subject_events) < window_size:
        return []

    sequences = []
    for i in range(len(subject_events) - window_size + 1):
        window = subject_events[i:i + window_size]
        event_probs = np.stack([e["probs"] for e in window])
        stats = calculate_event_stats(window)
        sequences.append({
            "sequence": event_probs,
            "stats": stats,
            "label": window[0]["osa_label"]
        })
    return sequences


class SequenceDataset(Dataset):
    """第二阶段序列数据集"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.FloatTensor(item["sequence"]),
            torch.FloatTensor(item["stats"]),
            torch.LongTensor([item["label"]])
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../../config.yaml")
    parser.add_argument("--epochs1", type=int, default=30, help="事件检测模型训练轮次")
    parser.add_argument("--epochs2", type=int, default=30, help="OSA诊断模型训练轮次")
    parser.add_argument("--save_path", default="models/two_stage/")
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--window_size", type=int, default=10, help="事件序列滑动窗口大小")
    args = parser.parse_args()

    # 初始化路径
    os.makedirs(args.save_path, exist_ok=True)
    config = load_config(resolve_dir(args.config, [Path.cwd(), project_root()]))
    anchors = [Path.cwd(), project_root()]
    train_dir = resolve_dir(args.train_dir or cfg_get(config, 'paths.signals_path'), anchors)
    val_dir = resolve_dir(args.val_dir or cfg_get(config, 'paths.signals_path'), anchors)

    # -------------------------
    # 第一阶段：事件检测模型训练
    # -------------------------
    print("=" * 60)
    print("开始第一阶段：事件检测模型训练")
    print("=" * 60)

    # 定义标签映射并验证
    event_labels = [
        "Hypopnea",
        "ObstructiveApnea",
        "normal"
    ]
    event_label_map = {name: i for i, name in enumerate(event_labels)}
    num_event_classes = len(event_labels)  # 现在类别数=3



    print(f"事件标签映射：{event_label_map}")
    valid_label_values = set(event_label_map.values())
    if valid_label_values != {0, 1, 2}:
        raise ValueError(f"标签映射异常！预期 {{0,1,2,3}}，实际 {valid_label_values}")
    print(f"有效标签值：{valid_label_values}")  # 应为 {0,1,2,3,4}

    # 加载数据
    train_events = load_event_data(train_dir, event_label_map)
    val_events = load_event_data(val_dir, event_label_map)

    if len(train_events) == 0:
        raise ValueError("训练集无有效事件！请检查数据标签")
    if len(val_events) == 0:
        raise ValueError("验证集无有效事件！请检查数据标签")

    # 计算标准化参数
    def calculate_mean_std(events):
        all_pixels = []
        for ev in events:
            all_pixels.extend(ev["data"].flatten())
        mean = np.mean(all_pixels) / 255.0
        std = np.std(all_pixels) / 255.0
        return mean, std

    train_mean, train_std = calculate_mean_std(train_events)
    print(f"训练数据标准化参数 - 均值: {train_mean:.4f}, 标准差: {train_std:.4f}")

    # 构建数据集
    train_dataset = SignalDatasetnew(
        train_events,
        classes=event_label_map,
        mean=train_mean,
        std=train_std
    )
    val_dataset = SignalDatasetnew(
        val_events,
        classes=event_label_map,
        mean=train_mean,
        std=train_std
    )

    # 在 two_stage_train.py 中修改 collate_fn
    def collate_fn(batch):
        imgs = []
        labels = []
        for item in batch:
            img, label = item  # 每个样本返回 (img, label)
            imgs.append(img)
            labels.append(label)
        # 拼接图像和标签（确保批次大小一致）
        return torch.stack(imgs), torch.cat(labels).squeeze()  # labels 变为 (batch_size,)

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    event_model = EventDetector(num_classes=num_event_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(event_model.parameters(), lr=5e-4, weight_decay=1e-4)

    # 训练事件检测模型
    best_event_f1 = 0.0
    for epoch in range(args.epochs1):
        print(f"\nEpoch {epoch + 1}/{args.epochs1}")
        print("-" * 50)

        train_loss, train_f1 = train_epoch(event_model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = eval_model(event_model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   F1: {val_f1:.4f}")

        if val_f1 > best_event_f1:
            best_event_f1 = val_f1
            save_path = os.path.join(args.save_path, "event_detector_best.pth")
            torch.save(event_model.state_dict(), save_path)
            print(f"保存最优事件检测模型到: {save_path} (F1: {best_event_f1:.4f})")

    # -------------------------
    # 第二阶段：OSA诊断模型训练
    # -------------------------
    print("\n" + "=" * 60)
    print("开始第二阶段：OSA诊断模型训练")
    print("=" * 60)

    # 加载最优事件检测模型
    event_model.load_state_dict(torch.load(os.path.join(args.save_path, "event_detector_best.pth")))
    event_model.eval()

    # 生成事件序列数据
    def generate_sequence_data(events):
        results = []
        with torch.no_grad():
            for ev in tqdm(events, desc="生成事件序列特征"):
                # 数据预处理（与训练时一致）
                img = torch.FloatTensor(ev["data"]).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                img = img / 255.0
                img = (img - train_mean) / train_std
                img = img.to(device)

                # 事件类型预测
                pred = F.softmax(event_model(img), dim=1).cpu().numpy()[0]

                # 生成OSA标签（请根据实际标签规则修改！）
                # 注意：当前为示例逻辑，需替换为真实标签获取方式
                osa_label = 1 if "apnea" in ev["subject_id"].lower() else 0

                results.append({
                    "probs": pred,
                    "timestamp": ev["timestamp"],
                    "subject_id": ev["subject_id"],
                    "end_time": ev["end_time"],
                    "osa_label": osa_label
                })

        # 按受试者生成滑动窗口序列
        subjects = list(set([e["subject_id"] for e in results]))
        sequences = []
        for subj in subjects:
            sequences.extend(create_event_sequences(results, subj, args.window_size))
        return sequences

    # 生成训练和验证序列
    train_sequences = generate_sequence_data(train_events)
    val_sequences = generate_sequence_data(val_events)

    # 序列数据验证
    print(f"\n生成训练序列数: {len(train_sequences)}")
    print(f"生成验证序列数: {len(val_sequences)}")

    if len(train_sequences) == 0:
        raise ValueError("未生成训练序列！请减小window_size或检查数据量")
    if len(val_sequences) == 0:
        raise ValueError("未生成验证序列！请减小window_size或检查数据量")

    # 构建序列数据集
    train_seq_dataset = SequenceDataset(train_sequences)
    val_seq_dataset = SequenceDataset(val_sequences)

    # 序列数据加载器
    def seq_collate(batch):
        sequences, stats, labels = zip(*batch)
        return (
            torch.stack(sequences),
            torch.stack(stats),
            torch.cat(labels)
        )

    seq_train_loader = DataLoader(
        train_seq_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=seq_collate,
        num_workers=0
    )
    seq_val_loader = DataLoader(
        val_seq_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=seq_collate,
        num_workers=0
    )

    # 初始化诊断模型
    osa_model = OSADiagnoser(num_diagnosis_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(osa_model.parameters(), lr=3e-4, weight_decay=1e-4)

    # 训练诊断模型
    best_osa_f1 = 0.0
    for epoch in range(args.epochs2):
        print(f"\nEpoch {epoch + 1}/{args.epochs2}")
        print("-" * 50)

        # 训练阶段
        osa_model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for seq, stats, labels in tqdm(seq_train_loader, desc="训练"):
            seq, stats, labels = seq.to(device), stats.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = osa_model(seq, stats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = total_loss / len(seq_train_loader)
        train_f1 = f1_score(all_labels, all_preds, average='weighted') if all_labels else 0.0

        # 验证阶段
        osa_model.eval()
        total_val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for seq, stats, labels in tqdm(seq_val_loader, desc="验证"):
                seq, stats, labels = seq.to(device), stats.to(device), labels.to(device)
                outputs = osa_model(seq, stats)
                total_val_loss += criterion(outputs, labels).item()

                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss = total_val_loss / len(seq_val_loader)
        val_f1 = f1_score(val_labels, val_preds, average='weighted') if val_labels else 0.0

        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   F1: {val_f1:.4f}")

        if val_f1 > best_osa_f1:
            best_osa_f1 = val_f1
            save_path = os.path.join(args.save_path, "osa_diagnoser_best.pth")
            torch.save(osa_model.state_dict(), save_path)
            print(f"保存最优OSA诊断模型到: {save_path} (F1: {best_osa_f1:.4f})")

    # 训练总结
    print("\n" + "=" * 60)
    print("两阶段模型训练完成！")
    print(f"事件检测最优F1: {best_event_f1:.4f}")
    print(f"OSA诊断最优F1: {best_osa_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()