# -*- coding: utf-8 -*-
"""
osa_end2end_events_ABC.py

三合一增强版（A/B/C）：
A) 数据层：支持滑窗步长 (stride) 与谱图增强 (SpecAugment)
B) 损失层：Class-Balanced Loss（可与 Focal 结合），缓解类别不均衡
C) 判决层：两阶段层级判决（Abnormal 判定 → Hyp vs OA），验证集自动搜索阈值

并新增：按受试者的交叉验证 (LOSO / KFold)；也可沿用固定 train/val 目录的单折训练。

依赖你的骨干模型：
    from src.models.osa_end2end import OSAEnd2EndModel
骨干输出 3 类 logits；不改内部结构。

运行示例：
1) 单折（沿用 train/val 目录）+ A/B/C 全开：
   python src/train/osa_end2end_events_ABC.py --train_dir data/processed/signals --val_dir data/processed/val \
        --epochs 30 --stride_train 5 --stride_val 10 --aug_spec \
        --use_cbl --cbl_beta 0.9999 --use_focal --focal_gamma 1.5

2) LOSO 交叉验证（data_root 下放全部 pickle；用 subject_id 划折）：
   python src/train/osa_end2end_events_ABC.py --cv_mode loso --data_root data/processed/all_subjects \
        --epochs 20 --stride_train 5 --stride_val 10 --aug_spec --use_cbl

3) KFold=5 交叉验证：
   python src/train/osa_end2end_events_ABC.py --cv_mode kfold --k 5 --data_root data/processed/all_subjects \
        --epochs 20 --stride_train 5 --stride_val 10 --aug_spec --use_cbl

保存：
- 每折都会在 save_dir 下保存 best pth，文件名包含折信息
- 也会打印/保存（可选）最佳阈值 tau_abn/tau_h/tau_oa
"""

import os
import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
from tqdm import tqdm

# 你的模型骨干
from src.models.osa_end2end import OSAEnd2EndModel

EVENT_CLASS_NAMES = ["Normal", "Hypopnea", "ObstructiveApnea"]
RAW2TRAIN = {
    # Normal
    "normal": 0, "none": 0, "background": 0, "noevent": 0, "negative": 0,
    # Hypopnea
    "hypopnea": 1, "hypopnoea": 1,
    # Obstructive / Mixed -> OA
    "obstructive apnea": 2, "obstructiveapnea": 2, "oa": 2,
    "mixed apnea": 2, "mixedapnea": 2, "ma": 2,
}
IGNORE_SET = {"central apnea", "centralapnea", "ca"}

# -------------------------
# 工具
# -------------------------

# ---------- 时间单位工具 ----------
def to_seconds(x, unit="sec", fs=None, hop_len=None):
    """
    把时间 x 转成秒：
      - unit='sec'：x 已是秒
      - unit='sample'：x 是样本点，需要 fs
      - unit='frame'：x 是帧索引，需要 hop_len（每帧样本数）和 fs
    """
    if unit == "sec":
        return float(x)
    elif unit == "sample":
        assert fs and fs > 0, "unit=sample 需要提供 fs"
        return float(x) / float(fs)
    elif unit == "frame":
        assert hop_len and fs and fs > 0, "unit=frame 需要 hop_len 与 fs"
        return float(x) * float(hop_len) / float(fs)
    else:
        raise ValueError(f"未知时间单位: {unit}")

def segment_label_from_events(seg_start, seg_end, events, overlap_thresh=0.2):
    """
    seg_start/seg_end 已是“秒”，events 里只统计 Hypopnea/OA 的时间重叠。
    返回三分类标签：0=Normal, 1=Hypopnea, 2=ObstructiveApnea
    """
    dur = max(1e-6, seg_end - seg_start)
    dur_h, dur_o = 0.0, 0.0

    for lab, ev_s, ev_e in events:  # lab: 0/1/2，仅传入窗口内事件即可
        s = max(seg_start, ev_s)
        t = min(seg_end,   ev_e)
        inter = max(0.0, t - s)
        if lab == 1:
            dur_h += inter
        elif lab == 2:
            dur_o += inter

    frac_h = min(1.0, max(0.0, dur_h / dur))
    frac_o = min(1.0, max(0.0, dur_o / dur))
    if max(frac_h, frac_o) >= overlap_thresh:
        return 1 if frac_h >= frac_o else 2, frac_h, frac_o
    return 0, frac_h, frac_o



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

def load_pickle_events(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def to_float_image(sig, force_hw=None):
    import torch as _t, numpy as _n
    x = _n.asarray(sig, _n.float32); x = _n.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if force_hw is not None and x.shape != tuple(force_hw):
        ten = _t.from_numpy(x)[None, None, ...]
        ten = F.interpolate(ten, size=force_hw, mode="bilinear", align_corners=False)
        x = ten.squeeze().numpy().astype(_n.float32)
    return x

class SingleSubjectSeq(Dataset):
    def __init__(self, val_dir, sid, img_size=64, seq_len=10):
        import pickle, os, numpy as np
        self.samples = []
        path = os.path.join(val_dir, f"{sid}.pickle")
        events = pickle.load(open(path, "rb"))
        imgs = []
        for ev in events:
            lab_raw = str(getattr(ev, "label", "none")).lower().strip()
            if lab_raw in IGNORE_SET: continue
            if lab_raw not in RAW2TRAIN: continue
            sig = getattr(ev, "signal", None)
            if sig is None: continue
            imgs.append(to_float_image(sig, force_hw=(img_size, img_size)))
        arr = np.stack(imgs, 0)
        mu, std = float(arr.mean()), float(arr.std()); std = std if std >= 1e-6 else 1e-3
        for i in range(0, len(imgs) - seq_len + 1):
            frames = []
            for j in range(seq_len):
                x = (imgs[i+j] - mu) / std; x = np.clip(x, -5.0, 5.0)
                frames.append(torch.from_numpy(x).float().unsqueeze(0))
            self.samples.append(torch.stack(frames, dim=0))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def ema_smooth(probs, alpha=0.6):
    if len(probs)==0: return probs
    out = np.zeros_like(probs); out[0] = probs[0]
    for i in range(1, len(probs)):
        out[i] = alpha * out[i-1] + (1 - alpha) * probs[i]
    return out

# -------------------------
# A) Dataset：stride + SpecAugment
# -------------------------
def spec_augment_torch(img_t: torch.Tensor,
                       time_mask_pct=0.1,
                       freq_mask_pct=0.1,
                       noise_std=0.01,
                       gain_db_range=(-2.0, 2.0)):
    """
    img_t: (1, H, W) or (T,1,H,W) 单帧增强时传 (1,H,W)
    简化版 SpecAugment：time/freq mask + 轻微高斯噪声 + 增益
    """
    x = img_t.clone()
    if x.dim() == 3:
        # (1,H,W)
        _, H, W = x.shape
        # 增益
        g = 10 ** (np.random.uniform(*gain_db_range) / 20.0)
        x = x * float(g)
        # 噪声
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        # time mask（沿 W 方向）
        tw = max(1, int(W * time_mask_pct))
        t0 = np.random.randint(0, max(1, W - tw + 1))
        x[:, :, t0:t0+tw] = 0.0
        # freq mask（沿 H 方向）
        fh = max(1, int(H * freq_mask_pct))
        f0 = np.random.randint(0, max(1, H - fh + 1))
        x[:, f0:f0+fh, :] = 0.0
        return x
    elif x.dim() == 4:
        # (T,1,H,W)：对每帧做轻量增强
        out = []
        for t in range(x.size(0)):
            out.append(spec_augment_torch(x[t], time_mask_pct, freq_mask_pct, noise_std, gain_db_range))
        return torch.stack(out, dim=0)
    else:
        return x

class EventSeqDataset(Dataset):
    """
    省内存版事件序列数据集：
    - 仅保存 (sid, start_idx) 索引；getitem 时按需取 [start : start+seq_len] 片段
    - 支持 stride、SpecAugment、按受试者 z-score
    - 窗口标签用“事件时间重叠率”判定：
        * 三分类：max(frac_h, frac_oa) >= overlap_thresh 时，取较大者（1=Hyp, 2=OA），否则 0=Normal
        * 二分类(binary_abnormal)：(frac_h+frac_oa) >= overlap_thresh → 1（异常），否则 0（正常）
    - 时间单位可配置：sec / sample / frame（需要 fs/hop_len）
    需要外部提供：
        - load_pickle_events(pickle_path)      # 返回事件列表
        - to_float_image(sig, force_hw=(H, W)) # 把二维信号转为 float32 图像
    事件对象 / 记录需要至少包含：.label / .signal / .start / .end
    """

    def __init__(self,
                 data_dir: str,
                 img_size: int = 64,
                 seq_len: int = 10,
                 train: bool = True,
                 debug_probe: bool = False,
                 stride: int = 1,
                 aug_spec: bool = False,
                 max_windows_per_subject: int = None,
                 overlap_thresh: float = 0.2,
                 three_class: bool = True,
                 # 时间单位设置
                 time_unit: str = "sec",  # 'sec' | 'sample' | 'frame'
                 fs: float = None,        # 采样率（当 unit 为 sample/frame 时需要）
                 hop_len: int = None      # 每帧样本数（当 unit 为 frame 时需要）
                 ):
        super().__init__()

        self.data_dir = data_dir
        self.img_size = int(img_size)
        self.seq_len = int(seq_len)
        self.train = bool(train)
        self.debug_probe = bool(debug_probe)
        self.stride = max(1, int(stride))
        self.aug_spec = bool(aug_spec) and self.train
        self.max_windows_per_subject = max_windows_per_subject

        self.overlap_thresh = float(overlap_thresh)
        self.three_class = bool(three_class)

        self.time_unit = str(time_unit).lower()
        self.fs = fs
        self.hop_len = hop_len

        # 数据缓存
        # sid -> list of (img(np.float32 HxW), cls:int, start_sec:float, end_sec:float)
        self.events_by_subject: dict[str, list[tuple]] = {}
        # sid -> (mu, std) 供 z-score
        self.subj_stats: dict[str, tuple[float, float]] = {}
        # 索引表：[(sid, start_idx), ...]
        self.index: list[tuple[str, int]] = []
        self.index2sid: list[str] = []  # 调试用

        self._scan_subjects()
        self._build_index()

        if self.debug_probe:
            # 用 per-event 标签粗略统计分布（不展开窗口）
            from collections import Counter
            cnt = Counter()
            for sid, evs in self.events_by_subject.items():
                for _, y, _, _ in evs:
                    cnt[y] += 1
            print(f"[Dataset] 事件级标签分布近似：{dict(cnt)} (0=Normal,1=Hypopnea,2=OA)")
            print(f"[Index] 窗口数：{len(self.index)} (seq_len={self.seq_len}, stride={self.stride})")

    # -------------------------
    # 工具：时间单位转换/重叠/增强
    # -------------------------
    @staticmethod
    def _to_seconds(x, unit="sec", fs=None, hop_len=None) -> float:
        """把时间 x 转换为秒。"""
        if unit == "sec":
            return float(x)
        elif unit == "sample":
            assert fs and fs > 0, "unit=sample 需要提供 fs"
            return float(x) / float(fs)
        elif unit == "frame":
            assert hop_len and fs and fs > 0, "unit=frame 需要 hop_len 与 fs"
            return float(x) * float(hop_len) / float(fs)
        else:
            raise ValueError(f"未知时间单位: {unit}")

    @staticmethod
    def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
        """区间重叠长度（秒）。"""
        return max(0.0, min(a1, b1) - max(a0, b0))

    def _spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        """简化版 SpecAugment：随机时间/频率遮罩 + 随机增益。x:(H,W)"""
        H, W = x.shape
        if np.random.rand() < 0.5:
            w = np.random.randint(1, max(2, W // 12))
            st = np.random.randint(0, max(1, W - w))
            x[:, st:st + w] = 0
        if np.random.rand() < 0.5:
            h = np.random.randint(1, max(2, H // 12))
            st = np.random.randint(0, max(1, H - h))
            x[st:st + h, :] = 0
        if np.random.rand() < 0.5:
            x.mul_(0.9 + 0.2 * np.random.rand())
        return x

    # -------------------------
    # 数据扫描与索引构建
    # -------------------------
    def _scan_subjects(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在：{self.data_dir}")

        pkl_files = sorted([fn for fn in os.listdir(self.data_dir) if fn.endswith(".pickle")])
        print(f"[Scan] 目录：{self.data_dir} | .pickle 文件数：{len(pkl_files)}")

        for fn in pkl_files:
            sid = os.path.splitext(fn)[0]
            pkl = os.path.join(self.data_dir, fn)
            try:
                events = load_pickle_events(pkl)
            except Exception as e:
                print(f"[warn] 读取失败：{pkl} → {e}")
                continue

            items = []
            imgs_tmp, labels_tmp = [], []

            for ev in events:
                lab_raw = str(getattr(ev, "label", "none")).lower().strip()
                if lab_raw in IGNORE_SET:
                    continue
                if lab_raw not in RAW2TRAIN:
                    continue
                cls = RAW2TRAIN[lab_raw]

                sig = getattr(ev, "signal", None)
                if sig is None or np.asarray(sig).ndim != 2:
                    continue

                # 时间统一为秒
                s_raw = float(getattr(ev, "start", 0.0))
                e_raw = float(getattr(ev, "end", s_raw))
                s_sec = self._to_seconds(s_raw, unit=self.time_unit, fs=self.fs, hop_len=self.hop_len)
                e_sec = self._to_seconds(e_raw, unit=self.time_unit, fs=self.fs, hop_len=self.hop_len)
                if e_sec <= s_sec:
                    e_sec = s_sec + 1e-3  # 防御性修正

                img = to_float_image(sig, force_hw=(self.img_size, self.img_size))  # np.float32(H,W)
                items.append((img, cls, s_sec, e_sec))
                imgs_tmp.append(img)
                labels_tmp.append(cls)

            if not items:
                print(f"[Scan] {fn}: 可用事件=0（可能全为 CA 或无效）")
                continue

            arr = np.stack(imgs_tmp, 0)
            mu, std = float(arr.mean()), float(arr.std())
            if not np.isfinite(std) or std < 1e-6:
                std = 1e-3

            self.subj_stats[sid] = (mu, std)
            self.events_by_subject[sid] = items

            from collections import Counter
            cnt = Counter(labels_tmp)
            print(f"[Scan] {fn}: 事件数={len(items)} | μ={mu:.4f} σ={std:.4f} | 分布={dict(cnt)}")

            # 释放中间数组
            del arr, imgs_tmp, labels_tmp

    def _build_index(self):
        """生成 (sid, start_idx) 索引，不复制图像数据."""
        for sid, lst in self.events_by_subject.items():
            L = len(lst)
            if L < self.seq_len:
                continue

            starts = list(range(0, L - self.seq_len + 1, self.stride))
            if self.train and self.max_windows_per_subject is not None and len(starts) > self.max_windows_per_subject:
                rng = np.random.default_rng()
                starts = rng.choice(starts, size=self.max_windows_per_subject, replace=False).tolist()

            for st in starts:
                self.index.append((sid, st))
                self.index2sid.append(sid)

        print(f"[MakeSeq] 生成序列索引：{len(self.index)} 条 (seq_len={self.seq_len}, stride={self.stride}) | subjects={len(self.events_by_subject)}")

    # -------------------------
    # PyTorch Dataset 接口
    # -------------------------
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sid, st = self.index[idx]
        mu, std = self.subj_stats.get(sid, (0.0, 1.0))
        evs = self.events_by_subject[sid]  # list of (img, cls, s_sec, e_sec)

        seq_frames = []

        # 1) 构建窗口帧并确定窗口时间范围（秒）
        seg_start = min(evs[k][2] for k in range(st, st + self.seq_len))
        seg_end   = max(evs[k][3] for k in range(st, st + self.seq_len))
        total_dur = max(1e-6, seg_end - seg_start)

        # 2) 统计窗口内 Hyp/OA 的时间重叠（只对窗口覆盖的这些事件统计即可）
        dur_hyp, dur_oa = 0.0, 0.0

        for k in range(st, st + self.seq_len):
            img_np, lab, ev_s, ev_e = evs[k]

            # z-score & 裁剪 & 可选增强
            x = (img_np - mu) / std
            x = np.clip(x, -5.0, 5.0)
            ten = torch.from_numpy(x).float()  # (H, W)
            if self.aug_spec:
                ten = self._spec_augment(ten)
            ten = ten.unsqueeze(0)  # (1, H, W)
            seq_frames.append(ten)

            inter = self._overlap(seg_start, seg_end, ev_s, ev_e)
            if lab == 1:
                dur_hyp += inter
            elif lab == 2:
                dur_oa += inter

        # 3) 重叠率→窗口标签
        frac_h = min(1.0, max(0.0, dur_hyp / total_dur))
        frac_o = min(1.0, max(0.0, dur_oa  / total_dur))
        if self.three_class:
            if max(frac_h, frac_o) >= self.overlap_thresh:
                label = 1 if frac_h >= frac_o else 2
            else:
                label = 0
        else:
            abn_frac = frac_h + frac_o
            label = 1 if abn_frac >= self.overlap_thresh else 0

        seq_tensor = torch.stack(seq_frames, dim=0)  # (T, 1, H, W)

        if self.debug_probe and idx < 2:
            xx = seq_tensor
            print(f"[DBG] sample#{idx}@{sid}: min={float(xx.min()):.3f}, max={float(xx.max()):.3f}, var={float(xx.var()):.6f}")
            print(f"[DBG] seg=({seg_start:.2f},{seg_end:.2f}) dur={total_dur:.2f} | frac_h={frac_h:.3f} frac_o={frac_o:.3f} -> y={label}")

        return seq_tensor, torch.tensor(label, dtype=torch.long), sid




# -------------------------
# B) Class-Balanced Loss（可与 Focal 叠加）
# -------------------------
class ClassBalancedLoss(nn.Module):
    """
    Cui et al., "Class-Balanced Loss Based on Effective Number of Samples"
    支持与 Focal 组合：loss = CB_weight * Focal(CE)
    """
    def __init__(self, samples_per_class, beta=0.9999, gamma=0.0, reduction='mean'):
        super().__init__()
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.reduction = reduction

        effective_num = 1.0 - np.power(self.beta, np.asarray(samples_per_class, np.float64))
        weights = (1.0 - self.beta) / np.maximum(effective_num, 1e-8)
        weights = weights / np.sum(weights) * len(samples_per_class)
        self.weight = torch.tensor(weights, dtype=torch.float32)  # on CPU; move in forward

    def forward(self, logits, targets):
        # CE per-sample
        ce = F.cross_entropy(logits, targets, weight=self.weight.to(logits.device), reduction='none')
        if self.gamma > 0:
            pt = torch.exp(-ce)
            loss = (1 - pt) ** self.gamma * ce
        else:
            loss = ce
        if self.reduction == 'mean':
            return loss.mean()
        return loss

# -------------------------
# 训练与评估（含 C：层级判决与阈值搜索）
# -------------------------
def train_epoch(model, loader, criterion, optimizer, device, probe=False):
    from sklearn.metrics import confusion_matrix, f1_score

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    total_loss = 0.0
    all_p, all_y = [], []
    did_probe = not probe

    for seq, y, _ in tqdm(loader, desc="训练(事件)"):
        seq = seq.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(seq)  # (B,C)  这里 C=3（Normal/Hyp/OA）
            # === 层级式损失：异常(非0) 的 BCE + 原分类损失 ===
            abn_target = (y != 0).long()
            p = F.softmax(out, dim=1)
            p_abn = (1.0 - p[:, 0]).clamp(1e-6, 1-1e-6)
            loss_abn = F.binary_cross_entropy(p_abn.unsqueeze(1), abn_target.float().unsqueeze(1))
            loss = criterion(out, y) + 0.5 * loss_abn

        if not did_probe:
            bx = seq.detach()
            print(f"[Probe] input var={bx.var().item():.6f}, min={bx.min().item():.3f}, max={bx.max().item():.3f}")
            stdlog = out.detach().float().std(dim=0).cpu().numpy().tolist()
            print(f"[Probe] logits std = {np.round(stdlog, 4)}")
            did_probe = True

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * seq.size(0)
        pred = out.argmax(dim=1)
        all_p.extend(pred.detach().cpu().numpy())
        all_y.extend(y.detach().cpu().numpy())

        # 及时释放
        del seq, y, out, pred

    # === 训练集指标汇总 ===
    avg_loss = total_loss / len(loader.dataset)
    f1w = f1_score(all_y, all_p, average="weighted", zero_division=0)

    # 动态确定出现过的标签，兼容二分类/三分类
    label_set = sorted(set(all_y) | set(all_p))
    cm = confusion_matrix(all_y, all_p, labels=label_set).astype(np.int64)
    cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    # 逐类 F1
    per_f1 = f1_score(all_y, all_p, labels=label_set, average=None, zero_division=0)

    # 友好打印（按已出现的标签顺序映射名称）
    name_map = {0: "Normal", 1: "Hypopnea", 2: "ObstructiveApnea"}
    names = [name_map.get(i, str(i)) for i in label_set]

    print("\n[Train] 本轮训练集混淆矩阵（行=真值 / 列=预测）")
    print("labels:", names)
    print(cm)
    print("[Train] 行归一化混淆矩阵：")
    import numpy as _np
    print(_np.round(cmn, 2))
    for n, f in zip(names, per_f1):
        print(f"  {n}: F1={f:.4f}")

    return avg_loss, f1w




@torch.no_grad()
def eval_epoch(model, loader, criterion, device,
               search_hier=True,  # 是否进行两阶段层级阈值搜索
               tau_grid_abn=np.linspace(0.3, 0.8, 11),
               tau_grid_cls=np.linspace(0.3, 0.8, 6)):
    model.eval()
    total_loss = 0.0
    ys, probs = [], []
    for seq, y, _ in tqdm(loader, desc="评估(事件)"):
        seq = seq.to(device)
        y = y.to(device)
        out = model(seq)  # (B,3)
        loss = criterion(out, y)
        total_loss += float(loss.item()) * seq.size(0)
        p = out.softmax(dim=1)  # (B,3)
        probs.append(p.cpu().numpy())
        ys.append(y.cpu().numpy())
    probs = np.concatenate(probs, axis=0)  # (N,3)
    ys = np.concatenate(ys, axis=0)        # (N,)

    if not search_hier:
        abn_prob = probs[:, 1] + probs[:, 2]
        best_f1, best_tau, best_pred = -1, 0.5, None
        for t in tau_grid_abn:
            pred = np.argmax(probs, axis=1).copy()
            pred[(abn_prob < t)] = 0  # 压回 Normal
            f1w = f1_score(ys, pred, average="weighted", zero_division=0)
            if f1w > best_f1:
                best_f1, best_tau, best_pred = f1w, float(t), pred
        tau_pack = {"tau_abn": best_tau, "tau_h": None, "tau_oa": None}
    else:
        # C) 两阶段层级判决：先判 Abnormal，再在 Abnormal 内用 class-specific 阈值分 Hyp/OA
        best_f1, best_pack, best_pred = -1, None, None
        for t_abn in tau_grid_abn:
            # 初筛是否异常
            abn_mask = (probs[:, 1] + probs[:, 2]) >= t_abn
            # 默认都先置为 Normal
            pred = np.zeros(len(ys), dtype=np.int64)
            # 在异常子集上再决定 Hyp vs OA
            for t_h in tau_grid_cls:
                for t_oa in tau_grid_cls:
                    sub_pred = pred.copy()
                    # 在异常子集上：
                    # 若 p_h >= t_h 且 p_h >= p_oa => Hyp；否则若 p_oa >= t_oa 且 p_oa > p_h => OA；
                    # 否则回退到 (p_h vs p_oa) argmax
                    ph = probs[:, 1]
                    po = probs[:, 2]
                    # 先给异常子集赋予 argmax(h,oa)
                    choice = np.where(ph >= po, 1, 2)
                    # 提升：满足各自阈值的更“自信”样本
                    choice[(ph >= t_h) & (ph >= po) & abn_mask] = 1
                    choice[(po >= t_oa) & (po >  ph) & abn_mask] = 2
                    sub_pred[abn_mask] = choice[abn_mask]

                    f1w = f1_score(ys, sub_pred, average="weighted", zero_division=0)
                    if f1w > best_f1:
                        best_f1 = f1w
                        best_pred = sub_pred
                        best_pack = {"tau_abn": float(t_abn), "tau_h": float(t_h), "tau_oa": float(t_oa)}

        tau_pack = best_pack if best_pack is not None else {"tau_abn": 0.5, "tau_h": 0.5, "tau_oa": 0.5}

    # 报告
    cm = confusion_matrix(ys, best_pred, labels=[0, 1, 2])
    cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    print(f"[Val] 最优阈值: {tau_pack} | F1(w)={best_f1:.4f}")
    print("原始混淆矩阵：\n", cm)
    print("归一化混淆矩阵：\n", np.round(cmn, 2))
    per = f1_score(ys, best_pred, average=None, labels=[0, 1, 2], zero_division=0)
    for i, n in enumerate(EVENT_CLASS_NAMES):
        print(f"  {n}: F1={per[i]:.4f}")

    return total_loss / len(loader.dataset), best_f1, tau_pack

# -------------------------
# 单折训练
# -------------------------
def run_single_fold(args, train_dir, val_dir, fold_name="single"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 数据
    ds_tr = EventSeqDataset(
        train_dir, img_size=args.img_size, seq_len=args.seq_len,
        train=True, debug_probe=True,
        stride=args.stride_train, aug_spec=args.aug_spec,
        max_windows_per_subject=None,
        overlap_thresh=args.overlap_thresh,
        three_class=not args.binary_abnormal
    )
    ds_va = EventSeqDataset(
        val_dir, img_size=args.img_size, seq_len=args.seq_len,
        train=False, debug_probe=True,
        stride=args.stride_val, aug_spec=False,
        overlap_thresh=args.overlap_thresh,
        three_class=not args.binary_abnormal
    )


    # 采样权重
    y_tr = np.array([y for _, y, _ in ds_tr])
    cls_cnt = np.bincount(y_tr, minlength=3)
    if any(c == 0 for c in cls_cnt):
        raise RuntimeError(f"训练集中存在缺类：{cls_cnt.tolist()}")
    print(f"[Train Class Count] {cls_cnt.tolist()}")

    if args.use_cbl:
        criterion = ClassBalancedLoss(samples_per_class=cls_cnt, beta=0.9999, gamma=1.5)
        print(f"[Loss] ClassBalancedLoss(beta={args.cbl_beta}, gamma={'%.2f'%criterion.gamma})")
    else:
        if args.use_focal:
            criterion = ClassBalancedLoss(samples_per_class=[1,1,1], beta=0.0, gamma=args.focal_gamma)  # 仅 focal
            print(f"[Loss] FocalOnly(gamma={args.focal_gamma})")
        else:
            criterion = nn.CrossEntropyLoss()
            print("[Loss] CrossEntropyLoss")

    # 采样器（仍保留，进一步稳住批次分布）
    w_per_cls = 1.0 / np.maximum(cls_cnt, 1)
    w_sample = w_per_cls[y_tr]
    sampler = WeightedRandomSampler(weights=w_sample, num_samples=len(w_sample), replacement=True)


    dl_tr = DataLoader(
        ds_tr, batch_size=args.batch_size, sampler=sampler, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available(),
        persistent_workers=False, prefetch_factor=2 if 2 > 0 else None
    )
    dl_va = DataLoader(
        ds_va, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available(),
        persistent_workers=False, prefetch_factor=2 if 2 > 0 else None
    )

    # 模型 & 优化
    model = OSAEnd2EndModel(img_channels=1, img_size=args.img_size, seq_len=args.seq_len, num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_f1, best_tau_pack = -1.0, {"tau_abn": 0.5, "tau_h": 0.5, "tau_oa": 0.5}
    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, f"osa_end2end_best_3cls_{fold_name}.pth")

    for ep in range(1, args.epochs + 1):
        print(f"\n[{fold_name}] Epoch {ep}/{args.epochs}\n" + "-" * 50)
        tr_loss, tr_f1 = train_epoch(model, dl_tr, criterion, optimizer, device, probe=(ep == 1 and args.debug))
        print(f"Train Loss: {tr_loss:.4f} | Train F1(w): {tr_f1:.4f}")

        va_loss, va_f1, tau_pack = eval_epoch(model, dl_va, criterion, device,
                                              search_hier=True,
                                              tau_grid_abn=np.linspace(args.tau_abn_min, args.tau_abn_max, args.tau_abn_steps),
                                              tau_grid_cls=np.linspace(args.tau_cls_min, args.tau_cls_max, args.tau_cls_steps))
        print(f"Val   Loss: {va_loss:.4f} | Val   F1(w): {va_f1:.4f} | tau*: {tau_pack}")

        scheduler.step(va_f1)
        if va_f1 > best_f1:
            best_f1, best_tau_pack = va_f1, tau_pack
            torch.save({"state_dict": model.state_dict(), "best_tau_pack": best_tau_pack}, best_path)
            print(f"[Save] {best_path} (F1: {best_f1:.4f}, tau={best_tau_pack})")

    # 保存阈值 JSON
    with open(best_path + ".tau.json", "w") as f:
        json.dump(best_tau_pack, f, indent=2)
    print(f"[Done] fold={fold_name} best_f1={best_f1:.4f} | save={best_path}")
    return best_f1, best_tau_pack, best_path

# -------------------------
# 交叉验证：LOSO / KFold
# -------------------------
def list_subjects(data_root):
    subs = []
    for fn in sorted(os.listdir(data_root)):
        if fn.endswith(".pickle"):
            subs.append(os.path.splitext(fn)[0])
    return subs

def materialize_split(data_root, sub_ids, dst_dir):
    """
    为给定 subject 列表把 .pickle “软分发”到 dst_dir（用硬链接/软链接/拷贝均可。
    这里用硬链接如果可能，失败则拷贝）
    """
    os.makedirs(dst_dir, exist_ok=True)
    for sid in sub_ids:
        src = os.path.join(data_root, f"{sid}.pickle")
        dst = os.path.join(dst_dir, f"{sid}.pickle")
        if os.path.exists(dst):
            continue
        try:
            os.link(src, dst)  # 硬链接（同一盘）
        except Exception:
            import shutil
            shutil.copy2(src, dst)

def run_cv(args):
    data_root = args.data_root
    subs = list_subjects(data_root)
    if len(subs) < 2:
        raise RuntimeError(f"CV 模式至少需要 2 个受试者，但只发现 {len(subs)}")

    results = []
    anchors = [Path.cwd(), project_root()]
    base_tmp = Path(resolve_dir(args.tmp_split_dir, anchors))
    base_tmp.mkdir(parents=True, exist_ok=True)

    if args.cv_mode.lower() == "loso":
        for i, test_sid in enumerate(subs):
            train_sids = [s for s in subs if s != test_sid]
            fold_dir = base_tmp / f"fold_loso_{test_sid}"
            tr_dir = fold_dir / "train"
            va_dir = fold_dir / "val"
            if tr_dir.exists():
                import shutil
                shutil.rmtree(tr_dir)
            if va_dir.exists():
                import shutil
                shutil.rmtree(va_dir)
            materialize_split(data_root, train_sids, str(tr_dir))
            materialize_split(data_root, [test_sid], str(va_dir))

            f1, tau_pack, path = run_single_fold(args, str(tr_dir), str(va_dir), fold_name=f"loso_{test_sid}")
            results.append({"fold": f"loso_{test_sid}", "f1": f1, "tau": tau_pack, "ckpt": path})

    elif args.cv_mode.lower() == "kfold":
        kf = KFold(n_splits=args.k, shuffle=True, random_state=42)
        for i, (tr_idx, va_idx) in enumerate(kf.split(subs), 1):
            tr_sids = [subs[j] for j in tr_idx]
            va_sids = [subs[j] for j in va_idx]
            fold_dir = base_tmp / f"fold_k{i}"
            tr_dir = fold_dir / "train"
            va_dir = fold_dir / "val"
            if tr_dir.exists():
                import shutil
                shutil.rmtree(tr_dir)
            if va_dir.exists():
                import shutil
                shutil.rmtree(va_dir)
            materialize_split(data_root, tr_sids, str(tr_dir))
            materialize_split(data_root, va_sids, str(va_dir))

            f1, tau_pack, path = run_single_fold(args, str(tr_dir), str(va_dir), fold_name=f"k{i}")
            results.append({"fold": f"k{i}", "f1": f1, "tau": tau_pack, "ckpt": path})
    else:
        raise ValueError(f"未知 cv_mode: {args.cv_mode}")

    # 汇总
    mean_f1 = float(np.mean([r["f1"] for r in results]))
    print("\n==== CV 汇总 ====")
    for r in results:
        print(f"{r['fold']:>10s} | F1={r['f1']:.4f} | tau={r['tau']} | ckpt={r['ckpt']}")
    print(f"平均 F1: {mean_f1:.4f}")

    # 存盘
    out_json = Path(args.save_dir) / f"cv_{args.cv_mode}.summary.json"
    with open(out_json, "w") as f:
        json.dump({"mode": args.cv_mode, "mean_f1": mean_f1, "folds": results}, f, indent=2)
    print(f"[Saved] {out_json}")

# -------------------------
# Main
# -------------------------
# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    # 运行模式
    ap.add_argument("--cv_mode", choices=["none", "loso", "kfold"], default="loso",
                    help="none=按 --train_dir / --val_dir 单折；loso/kfold=按受试者做交叉验证")
    ap.add_argument("--data_root", type=str, default="/Users/liyuxiang/Downloads/sleep-apnea-main/data/processed/all_subjects",
                    help="LOSO 模式下所有受试者的 .pickle 根目录")
    ap.add_argument("--k", type=int, default=5, help="KFold 折数")
    ap.add_argument("--tmp_split_dir", type=str, default="runs/tmp_cv_splits", help="CV 中间划分输出目录")
    ap.add_argument("--ema_alpha", type=float, default=0.6)
    # 数据路径（单折模式）
    ap.add_argument("--train_dir", default="/Users/liyuxiang/Downloads/sleep-apnea-main/data/processed/signals")
    ap.add_argument("--val_dir",   default="/Users/liyuxiang/Downloads/sleep-apnea-main/data/processed/val")

    # 数据与模型
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--seq_len",  type=int, default=10)
    ap.add_argument("--stride_train", type=int, default=5)
    ap.add_argument("--stride_val",   type=int, default=5)   # ✅ 验证步长缩短为5
    ap.add_argument("--aug_spec", action="store_true", help="训练时启用 SpecAugment")
    ap.add_argument("--epochs",   type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)

    # 损失相关
    ap.add_argument("--use_cbl", action="store_true", help="启用 Class-Balanced Loss")
    ap.add_argument("--cbl_beta", type=float, default=0.9999)
    ap.add_argument("--use_focal", action="store_true")
    ap.add_argument("--focal_gamma", type=float, default=1.5)

    # 优化器
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--subject_id", default=None)

    # 两阶段阈值搜索范围
    ap.add_argument("--tau_abn_min", type=float, default=0.3)
    ap.add_argument("--tau_abn_max", type=float, default=0.8)
    ap.add_argument("--tau_abn_steps", type=int, default=11)
    ap.add_argument("--tau_cls_min", type=float, default=0.3)
    ap.add_argument("--tau_cls_max", type=float, default=0.8)
    ap.add_argument("--tau_cls_steps", type=int, default=6)

    # 其他
    ap.add_argument("--save_dir", default="models/end2end")
    ap.add_argument("--debug", action="store_true")

    ap.add_argument("--overlap_thresh", type=float, default=0.2, help="重叠率阈值")
    ap.add_argument("--binary_abnormal", action="store_true", help="若指定，则窗口标签二分类(0/1)")

    args = ap.parse_args()

    if args.cv_mode in ["loso", "kfold"]:
        # 交叉验证不需要 subject_id
        run_cv(args)
    else:
        # 单折/单受试者场景才可能用 subject_id
        # 如果你的逻辑要“指定某个受试者”才生效，就在这里检查：
        # if args.subject_id is None: raise ValueError("单受试者流程需要 --subject_id")
        run_single_fold(args, args.train_dir, args.val_dir, fold_name="single")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SingleSubjectSeq(args.val_dir, args.subject_id, img_size=args.img_size, seq_len=args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(args.event_ckpt, map_location=device, weights_only=False)
    model = OSAEnd2EndModel(img_channels=1, img_size=args.img_size, seq_len=args.seq_len, num_classes=3).to(device)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=True);
    model.eval()

    all_p = []
    with torch.no_grad():
        for seq in dl:
            seq = seq.to(device)
            out = model(seq)
            p = out.softmax(dim=1).cpu().numpy()
            all_p.append(p)
    probs = np.concatenate(all_p, axis=0)
    if args.ema_alpha > 0:
        probs = ema_smooth(probs, alpha=args.ema_alpha)

    # 无监督分位阈值初始化 + 小范围微调
    abn = probs[:, 1] + probs[:, 2]
    base = float(np.quantile(abn, 0.90))  # 90分位
    grid_abn = np.clip(np.linspace(base - 0.1, base + 0.1, 9), 0.05, 0.95)
    grid_cls = np.linspace(0.35, 0.75, 9)

    # 简单一致性目标：让异常比例接近分位估计；内部 Hyp/OA 用概率 argmax 与阈值一致率最大
    def score_triplet(t_abn, t_h, t_oa):
        pred = np.zeros(len(probs), dtype=np.int64)
        mask = abn >= t_abn
        ph, po = probs[:, 1], probs[:, 2]
        choice = np.where(ph >= po, 1, 2)
        choice[(ph >= t_h) & (ph >= po) & mask] = 1
        choice[(po >= t_oa) & (po > ph) & mask] = 2
        pred[mask] = choice[mask]
        # 一致性分：越稀疏越好，且内部选择稳定
        sparsity = 1.0 - (mask.mean())  # 少触发略加分
        stability = (choice[mask] == np.where(ph[mask] >= po[mask], 1, 2)).mean() if mask.any() else 1.0
        return 0.4 * sparsity + 0.6 * stability, {"tau_abn": float(t_abn), "tau_h": float(t_h), "tau_oa": float(t_oa)}

    best_sc, best_tau = -1, {"tau_abn": 0.5, "tau_h": 0.5, "tau_oa": 0.5}
    for ta in grid_abn:
        for th in grid_cls:
            for to in grid_cls:
                sc, tp = score_triplet(ta, th, to)
                if sc > best_sc: best_sc, best_tau = sc, tp

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(best_tau, f, indent=2)
    print(f"[OK] saved subject-specific tau → {args.out_json} | {best_tau}")


if __name__ == "__main__":
    main()
