# -*- coding: utf-8 -*-
import os
import argparse
import pickle
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

# ===== 模型导入（保持与你工程一致）=====
from src.models.osa_end2end import OSAEnd2EndModel
try:
    from src.models.wgan_gp import WGANGP
except Exception:
    WGANGP = None


# =========================
# 工具函数
# =========================
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


def to_float_tensor(sig, force_hw=None):
    """
    保留浮点动态范围，不再映射到uint8。
    返回 np.float32 的二维数组。
    """
    x = np.asarray(sig, np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if force_hw is not None and x.shape != tuple(force_hw):
        ten = torch.from_numpy(x)[None, None, ...]  # (1,1,H,W)
        ten = F.interpolate(ten, size=force_hw, mode="bilinear", align_corners=False)
        x = ten.squeeze().numpy().astype(np.float32)
    return x


# =========================
# 数据集
# =========================
class OSAEnd2EndDataset(Dataset):
    """
    二分类：normal(0) vs obstructiveapnea(1)
    - 保留浮点谱图
    - 每 subject（每个 .pickle 文件）做 z-score 标准化
    - 窗口多数投票作为序列标签
    - 训练时加轻微时间抖动（减弱相邻窗口强相关）
    """
    LABEL_ALIAS = {
        "obstructiveapnea": 1,
        "obstructive_apnea": 1,
        "oa": 1,

        "normal": 0,
        "noapnea": 0, "none": 0, "baseline": 0, "neg": 0, "negative": 0,
    }

    def __init__(self, data_dirs, seq_len=10, img_size=64,
                 train=True, debug_probe=True):
        self.seq_len = seq_len
        self.img_size = img_size
        self.train = train
        self.debug_probe = debug_probe

        # 先按 subject 聚合所有事件 → 计算 μ/σ
        self.events_by_subject = {}  # sid -> list[(img_float, label_int)]
        self.subj_stats = {}         # sid -> (mu, std)
        self._scan_and_collect(data_dirs)

        # 滑窗生成序列（训练态加入起点抖动）
        self.data = self._make_sequences()

        if len(self.data) == 0:
            hint = [
                "[致命] 数据集为空：没有生成任何序列。",
                "排查建议：",
                " - 路径是否写对：signals（复数）而不是 signal（单数）",
                " - 目录下是否存在 .pickle 文件",
                " - pickle 中 ev.label 是否能映射到 {normal, obstructiveapnea, ...}",
                " - ev.signal 是否为 2D 数组，且不是 NaN/Inf",
            ]
            raise RuntimeError("\n".join(hint))

        # 用于 __getitem__ 时快速查询 subject μ/σ
        self.index2sid = [sid for sid, _ in self.data]  # 保存每条样本对应的 subject id
        self.data = [v for _, v in self.data]           # 仅剩 (seq_imgs(list[np.float32]), label_int)

        # 打印分布
        if self.debug_probe:
            hist = Counter([y for _, y in self.data])
            print(f"[Dataset] sequence label hist: {hist}")

    def _scan_and_collect(self, data_dirs):
        total_files, total_events = 0, 0
        for d in data_dirs:
            print(f"[Scan] 目录：{d}")
            if (not d) or (not os.path.exists(d)):
                print(f"[warn] 数据目录不存在，跳过：{d}")
                continue
            pkl_list = sorted([fn for fn in os.listdir(d) if fn.endswith(".pickle")])
            print(f"[Scan] 找到 .pickle 文件：{len(pkl_list)} 个")
            for fn in pkl_list:
                total_files += 1
                pkl = os.path.join(d, fn)
                subject_id = os.path.splitext(fn)[0]

                try:
                    with open(pkl, "rb") as f:
                        events = pickle.load(f)
                except Exception as e:
                    print(f"[warn] 读取失败：{pkl} → {e}")
                    continue

                # 先把所有可用的谱图缓存下来，用于该 subject 的 μ/σ
                imgs, lbls = [], []
                mapped_labels = 0
                for ev in events:
                    total_events += 1
                    raw_label = str(getattr(ev, "label", "none")).strip()
                    key = raw_label.lower().replace("_", "").replace("-", "")
                    if key in self.LABEL_ALIAS:
                        cls = self.LABEL_ALIAS[key]
                        mapped_labels += 1
                    else:
                        continue
                    sig = getattr(ev, "signal", None)
                    if sig is None or (np.asarray(sig).ndim != 2):
                        continue
                    img = to_float_tensor(sig, force_hw=(self.img_size, self.img_size))  # float32, HxW
                    imgs.append(img)
                    lbls.append(cls)

                if len(imgs) == 0:
                    print(f"[Scan] {fn}: 映射标签数={mapped_labels}, 保留事件=0")
                    continue

                arr = np.stack(imgs, 0)  # (N,H,W)
                mu, std = float(arr.mean()), float(arr.std())
                if not np.isfinite(std) or std < 1e-6:
                    std = 1e-3
                self.subj_stats[subject_id] = (mu, std)
                self.events_by_subject[subject_id] = list(zip(imgs, lbls))
                print(f"[Scan] {fn}: 映射标签数={mapped_labels}, 保留事件={len(imgs)}, μ={mu:.4f}, σ={std:.4f}")

        if total_events == 0:
            print("[warn] 未在任何目录中解析到有效事件。")

    def _make_sequences(self):
        seq_data = []  # list[(sid, (list[np.float32], label_int))]
        for sid, lst in self.events_by_subject.items():
            if len(lst) < self.seq_len:
                continue
            imgs, lbls = zip(*lst)
            L = len(lst)
            for i in range(0, L - self.seq_len + 1):
                start = i
                if self.train and (L - self.seq_len > 5):
                    start = max(0, min(i + np.random.randint(-2, 3), L - self.seq_len))
                window_imgs = imgs[start:start + self.seq_len]
                window_lbls = lbls[start:start + self.seq_len]
                # 多数投票
                window_label = Counter(window_lbls).most_common(1)[0][0]
                seq_data.append((sid, (list(window_imgs), int(window_label))))

        print(f"数据加载完成：{len(seq_data)} 条序列 (seq_len={self.seq_len}), 来自 {len(self.events_by_subject)} 个subject, 总文件数={len(self.events_by_subject)}")
        # 打印标签统计
        seq_hist = Counter([v[1] for _, (_, v) in enumerate([(sid, (imgs, y)) for sid, (imgs, y) in seq_data])])
        # 上面一行是防止快排动作用的展开写法；等价于直接统计：
        seq_hist = Counter([y for _, (_, y) in seq_data])
        print(f"二分类标签统计: 0=normal({seq_hist.get(0,0)}) | 1=ObstructiveApnea({seq_hist.get(1,0)})")
        return seq_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sid = self.index2sid[idx]
        (seq_imgs, label) = self.data[idx]  # list[np.float32], int
        mu, std = self.subj_stats.get(sid, (0.0, 1.0))

        frames = []
        for img in seq_imgs:
            x = (img - mu) / std
            x = np.clip(x, -5.0, 5.0)
            ten = torch.from_numpy(x).float().unsqueeze(0)  # (1,H,W)
            frames.append(ten)
        seq_tensor = torch.stack(frames, dim=0)  # (T,1,H,W)

        if self.debug_probe and idx < 3:
            x = seq_tensor.detach()
            print(f"[DBG] sample#{idx}@{sid}: min={float(x.min()):.4f}, max={float(x.max()):.4f}, var={float(x.var()):.6f}")
        return seq_tensor, torch.tensor(label, dtype=torch.long)


# =========================
# 损失
# =========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        return loss


# =========================
# 评估与训练循环
# =========================
def print_confusion(y_true, y_pred, names, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    print("\n" + "=" * 50)
    print(f"{title} - 混淆矩阵")
    print("=" * 50)
    for i, n in enumerate(names):
        print(f"  索引 {i} -> {n}")
    print("\n原始混淆矩阵：")
    print(cm)
    print("\n归一化混淆矩阵（保留2位小数）：")
    print(np.round(cmn, 2))
    print("=" * 50 + "\n")


def train_epoch(model, loader, criterion, optim, device, first_probe=True):
    model.train()
    total_loss = 0.0
    all_p, all_y = [], []
    did_probe = not first_probe

    for seq, y in tqdm(loader, desc="训练"):
        # (B,T,1,H,W) → 模型期望
        seq = seq.to(device, non_blocking=True)
        y = y.view(-1).to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        out = model(seq)  # (B,2)

        if not did_probe:
            bx = seq.detach()
            print(f"[Probe][input] batch var={bx.var().item():.6f}, "
                  f"min={bx.min().item():.4f}, max={bx.max().item():.4f}")
        loss = criterion(out, y)
        loss.backward()

        if not did_probe:
            gnorm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    gnorm += float(g.norm().item() ** 2)
            gnorm = gnorm ** 0.5
            logits_std = out.detach().float().std(dim=0).cpu().numpy().tolist()
            print(f"[Probe] grad norm={gnorm:.6f}")
            print(f"[Probe] logits std across batch = {np.round(logits_std, 6)}")
            did_probe = True

        optim.step()

        total_loss += float(loss.item()) * seq.size(0)
        pred = out.argmax(dim=1)
        all_p.extend(pred.cpu().numpy())
        all_y.extend(y.cpu().numpy())

    return total_loss / len(loader.dataset), f1_score(all_y, all_p, average="weighted")


def eval_epoch(model, loader, criterion, device, names, title, tau=0.5):
    model.eval()
    total_loss = 0.0
    all_p, all_y = [], []
    with torch.no_grad():
        for seq, y in tqdm(loader, desc="评估"):
            seq = seq.to(device, non_blocking=True)
            y = y.view(-1).to(device, non_blocking=True)
            out = model(seq)
            loss = criterion(out, y)
            total_loss += float(loss.item()) * seq.size(0)

            # 阈值化（对 OA=1）
            proba = out.softmax(dim=1)[:, 1]
            pred = (proba > tau).long()

            all_p.extend(pred.cpu().numpy())
            all_y.extend(y.cpu().numpy())
    print_confusion(all_y, all_p, names, title)
    try:
        per = f1_score(all_y, all_p, average=None)
        for i, (n, v) in enumerate(zip(names, per)):
            print(f"  {n}: {v:.4f}")
    except Exception:
        pass
    return total_loss / len(loader.dataset), f1_score(all_y, all_p, average="weighted")


# =========================
# 主函数
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="/Users/liyuxiang/Downloads/sleep-apnea-main/data/processed/signals",
                        help="训练数据目录（pickle 事件）")
    parser.add_argument("--val_dir",   default="/Users/liyuxiang/Downloads/sleep-apnea-main/data/processed/val",
                        help="验证数据目录（pickle 事件）")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--use_focal", action="store_true", help="使用FocalLoss代替交叉熵")
    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--tau", type=float, default=0.5, help="验证/测试时OA(1)的概率阈值")
    parser.add_argument("--save_dir", default="models/end2end")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    anchors = [Path.cwd(), project_root()]
    train_dir = resolve_dir(args.train_dir, anchors)
    val_dir   = resolve_dir(args.val_dir, anchors)
    print(f"加载原始/增强训练数据：{train_dir}")
    print(f"验证数据：{val_dir}")

    # 数据集
    train_ds = OSAEnd2EndDataset([train_dir], seq_len=args.seq_len,
                                 img_size=args.img_size, train=True,  debug_probe=True)
    val_ds   = OSAEnd2EndDataset([val_dir],   seq_len=args.seq_len,
                                 img_size=args.img_size, train=False, debug_probe=True)

    # 类别分布 & 采样器
    train_labels = np.array([y for _, y in train_ds.data])
    val_labels   = np.array([y for _, y in val_ds.data])
    print("Train label counts:", Counter(train_labels.tolist()))
    print(" Val  label counts:", Counter(val_labels.tolist()))

    # 按类均衡的采样权重
    class_sample_count = np.bincount(train_labels, minlength=2)  # [n0, n1]
    if any(c == 0 for c in class_sample_count):
        raise RuntimeError(f"训练集中存在缺类：{class_sample_count.tolist()}。请检查标签映射/数据来源。")
    weights_per_class = 1.0 / np.maximum(class_sample_count, 1)
    sample_weights = weights_per_class[train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # 模型
    model = OSAEnd2EndModel(img_channels=1, img_size=args.img_size,
                            seq_len=args.seq_len, num_classes=2).to(device)

    # 损失：可选 focal / 交叉熵（注意：已用采样均衡，loss权重非必需）
    ce_weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    if args.use_focal:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=ce_weights)
        print(f"[Loss] 使用FocalLoss(gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=ce_weights)
        print("[Loss] 使用加权交叉熵（当前均等权；已由Sampler均衡采样）")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                           patience=5, verbose=True)

    class_names = ["normal", "ObstructiveApnea"]
    best_f1 = -1.0
    best_path = os.path.join(args.save_dir, "osa_end2end_best_2class.pth")

    for ep in range(args.epochs):
        print(f"\nEpoch {ep+1}/{args.epochs}\n" + "-" * 50)
        tr_loss, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, device, first_probe=(ep == 0))
        print(f"Train Loss: {tr_loss:.4f} | Train F1 (weighted): {tr_f1:.4f}")

        va_loss, va_f1 = eval_epoch(model, val_loader, criterion, device, class_names,
                                    f"Epoch {ep+1} 验证集", tau=args.tau)
        print(f"Val   Loss: {va_loss:.4f} | Val   F1 (weighted): {va_f1:.4f}")

        scheduler.step(va_f1)
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), best_path)
            print(f"保存最优模型到: {best_path} (F1: {best_f1:.4f})")

    # 最终评估
    if os.path.exists(best_path):
        sd = torch.load(best_path, map_location=device)
        try:
            model.load_state_dict(sd, strict=True)
        except TypeError:
            model.load_state_dict(sd)
        print("\n" + "=" * 60)
        print("训练结束，最优模型最终评估")
        print("=" * 60)
        va_loss, va_f1 = eval_epoch(model, val_loader, criterion, device, class_names,
                                    "最优模型验证集", tau=args.tau)
        print(f"最优模型最终加权F1: {va_f1:.4f}")

    print("\n" + "=" * 60)
    print("端到端OSA二分类训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
