# -*- coding: utf-8 -*-
"""
eval_multi_ckpts.py

用途：
- 对多个已训练权重（single / 多个 loso_*）在同一验证目录上进行评估。
- 支持两阶段（Abnormal->Hyp/OA）阈值判决，阈值优先从 <ckpt>.tau.json 读取，否则用默认/网格搜索。
- 支持对多个模型做 softmax 概率平均的集成评估。

依赖：
- 复用训练时的骨干与数据集：
    from src.models.osa_end2end import OSAEnd2EndModel
    from src.train.osa_end2end_events_ABC import EventSeqDataset
"""

import os
import json
import argparse
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset
from tqdm import tqdm

# ---- 复用你的模型与数据集 ----
from src.models.osa_end2end import OSAEnd2EndModel


EVENT_CLASS_NAMES = ["Normal", "Hypopnea", "ObstructiveApnea"]



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
    x = np.asarray(sig, np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if force_hw is not None and x.shape != tuple(force_hw):
        ten = torch.from_numpy(x)[None, None, ...]
        ten = F.interpolate(ten, size=force_hw, mode="bilinear", align_corners=False)
        x = ten.squeeze().numpy().astype(np.float32)
    return x

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


def load_model(ckpt_path: str, img_size: int, seq_len: int, num_classes: int = 3, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OSAEnd2EndModel(img_channels=1, img_size=img_size, seq_len=seq_len, num_classes=num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    model.eval()
    # tau json（可选）
    tau_json = {"tau_abn": 0.5, "tau_h": 0.5, "tau_oa": 0.5}
    tau_path = ckpt_path + ".tau.json"
    if os.path.exists(tau_path):
        try:
            with open(tau_path, "r") as f:
                tj = json.load(f)
            # 容错：缺哪个就用默认
            for k in ["tau_abn", "tau_h", "tau_oa"]:
                if k in tj and tj[k] is not None:
                    tau_json[k] = float(tj[k])
        except Exception as e:
            print(f"[warn] 读取阈值失败 {tau_path}: {e}，改用默认 {tau_json}")
    return model, tau_json


@torch.no_grad()
def collect_probs(model, loader, device):
    probs, ys = [], []
    for seq, y, _ in tqdm(loader, desc="Forward"):
        seq = seq.to(device)
        out = model(seq)              # (B,3)
        p = F.softmax(out, dim=1)     # (B,3)
        probs.append(p.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs, axis=0), np.concatenate(ys, axis=0)  # (N,3), (N,)


def hier_predict_from_probs(probs, tau_abn=0.5, tau_h=0.5, tau_oa=0.5):
    """
    两阶段层级判决：
      1) 先判断是否 Abnormal: (p_hyp+p_oa) >= tau_abn
      2) 在异常子集上：
             if p_h >= tau_h 且 p_h >= p_oa -> Hyp
        elif p_oa >= tau_oa 且 p_oa >  p_h  -> OA
        else 取 argmax(h, oa)
      正常子集 -> Normal
    """
    ph, po = probs[:, 1], probs[:, 2]
    abn_mask = (ph + po) >= tau_abn
    pred = np.zeros(len(ph), dtype=np.int64)
    if abn_mask.any():
        choice = np.where(ph >= po, 1, 2)
        choice[(ph >= tau_h) & (ph >= po)] = 1
        choice[(po >= tau_oa) & (po > ph)] = 2
        pred[abn_mask] = choice[abn_mask]
    return pred


def search_best_hier_thresholds(ys, probs, 
                                grid_abn=np.linspace(0.3, 0.8, 11),
                                grid_cls=np.linspace(0.3, 0.8, 6)):
    best_f1, best_pack, best_pred = -1.0, None, None
    ph, po = probs[:, 1], probs[:, 2]
    for t_abn in grid_abn:
        abn_mask = (ph + po) >= t_abn
        for t_h in grid_cls:
            for t_oa in grid_cls:
                pred = hier_predict_from_probs(probs, t_abn, t_h, t_oa)
                f1w = f1_score(ys, pred, average="weighted", zero_division=0)
                if f1w > best_f1:
                    best_f1, best_pack, best_pred = f1w, {"tau_abn": float(t_abn), "tau_h": float(t_h), "tau_oa": float(t_oa)}, pred
    return best_f1, best_pack, best_pred


def eval_one_model(name, model, tau_json, loader, device, do_search=False,
                   grid_abn=np.linspace(0.3, 0.8, 11), grid_cls=np.linspace(0.3, 0.8, 6)):
    probs, ys = collect_probs(model, loader, device)
    if do_search:
        f1w, tau_pack, pred = search_best_hier_thresholds(ys, probs, grid_abn, grid_cls)
    else:
        pred = hier_predict_from_probs(probs, tau_json["tau_abn"], tau_json["tau_h"], tau_json["tau_oa"])
        f1w = f1_score(ys, pred, average="weighted", zero_division=0)
        tau_pack = tau_json

    cm = confusion_matrix(ys, pred, labels=[0, 1, 2])
    cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    print(f"\n==== [{name}] ====")
    print(f"阈值: {tau_pack} | F1(w)={f1w:.4f}")
    print("混淆矩阵（行=真值 / 列=预测）:\n", cm)
    print("行归一化：\n", np.round(cmn, 2))
    per = f1_score(ys, pred, average=None, labels=[0, 1, 2], zero_division=0)
    for i, n in enumerate(EVENT_CLASS_NAMES):
        print(f"  {n}: F1={per[i]:.4f}")

    return {
        "name": name,
        "f1_weighted": float(f1w),
        "tau": tau_pack,
        "cm": cm.tolist(),
        "cm_norm": cmn.tolist(),
    }, probs, ys


def main():
    ap = argparse.ArgumentParser()
    # 数据
    ap.add_argument("--eval_dir", type=str, default="/Users/liyuxiang/Downloads/sleep-apnea-main/data/processed/t",
                    help="验证用 .pickle 目录（与训练一致的处理格式）")
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--time_unit", type=str, default="sec", choices=["sec", "sample", "frame"])
    ap.add_argument("--fs", type=float, default=None)
    ap.add_argument("--hop_len", type=int, default=None)
    # 模型路径（可多次传入）
    ap.add_argument("--single_ckpt", type=str, default="/Users/liyuxiang/Downloads/sleep-apnea-main/src/train/models/end2end/osa_end2end_best_3cls_single.pth")
    ap.add_argument("--loso_9995_ckpt", type=str, default="/Users/liyuxiang/Downloads/sleep-apnea-main/src/train/models/end2end/osa_end2end_best_3cls_loso_00000995.pth")  # 00000995
    ap.add_argument("--loso_9999_ckpt", type=str, default="/Users/liyuxiang/Downloads/sleep-apnea-main/src/train/models/end2end/osa_end2end_best_3cls_loso_00000999.pth")  # 00000999
    ap.add_argument("--loso_0000_ckpt", type=str, default="/Users/liyuxiang/Downloads/sleep-apnea-main/src/train/models/end2end/osa_end2end_best_3cls_loso_00001000.pth")  # 00001000
    ap.add_argument("--loso_1008_ckpt", type=str, default="/Users/liyuxiang/Downloads/sleep-apnea-main/src/train/models/end2end/osa_end2end_best_3cls_loso_00001008.pth")  # 00001008
    ap.add_argument("--loso_1131_ckpt", type=str, default="/Users/liyuxiang/Downloads/sleep-apnea-main/src/train/models/end2end/osa_end2end_best_3cls_loso_00001131.pth")  # 00001000
    # 评估选项
    ap.add_argument("--search_tau", action="store_true",
                    help="忽略各自 tau.json，统一在 eval_dir 上网格搜索 tau（更公平对比）")
    ap.add_argument("--ensemble", action="store_true",
                    help="对四个模型做 softmax 概率平均并评估")
    ap.add_argument("--ensemble_tau", choices=["avg", "search"], default="search",
                    help="集成阈值使用四个 tau 的平均(avg) 还是重新搜索(search)")
    ap.add_argument("--out_csv", type=str, default="eval_multi_ckpts_summary.csv")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 验证集（不做增强，按受试者 z-score；窗口标签规则与训练一致）
    ds_val = EventSeqDataset(
        data_dir=args.eval_dir,
        img_size=args.img_size,
        seq_len=args.seq_len,
        train=False,
        debug_probe=True,
        stride=args.stride,
        aug_spec=False,
        overlap_thresh=0.2,          # 与训练保持一致
        three_class=True,            # 三分类
        time_unit=args.time_unit,
        fs=args.fs,
        hop_len=args.hop_len
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    # 待评估模型集合
    ckpt_items = [
        ("single",      args.single_ckpt),
        ("loso_00000995", args.loso_9995_ckpt),
        ("loso_00000999", args.loso_9999_ckpt),
        ("loso_00001000", args.loso_0000_ckpt),
        ("loso_00001008", args.loso_0000_ckpt),
        ("loso_00001131", args.loso_0000_ckpt),

    ]

    results = []
    all_probs = []
    all_taus = []

    # 分别评估
    for name, path in ckpt_items:
        if not os.path.exists(path):
            print(f"[warn] 路径不存在，跳过：{name} -> {path}")
            continue
        model, tau_json = load_model(path, args.img_size, args.seq_len, 3, device)
        res, probs, ys = eval_one_model(
            name, model, tau_json, dl_val, device,
            do_search=args.search_tau
        )
        results.append(res)
        all_probs.append(probs)
        all_taus.append(tau_json)

    # 集成（可选）
    if args.ensemble and len(all_probs) >= 2:
        probs_ens = np.mean(np.stack(all_probs, axis=0), axis=0)  # (N,3)
        # 阈值策略
        if args.ensemble_tau == "avg" and not args.search_tau:
            # 用各自 tau 的平均（仅当单模不是统一搜索时，这样更一致）
            tau_pack = {
                "tau_abn": float(np.mean([t["tau_abn"] for t in all_taus])),
                "tau_h":   float(np.mean([t["tau_h"] for t in all_taus])),
                "tau_oa":  float(np.mean([t["tau_oa"] for t in all_taus])),
            }
            pred = hier_predict_from_probs(probs_ens, **tau_pack)
            f1w = f1_score(ys, pred, average="weighted", zero_division=0)
        else:
            # 对集成概率再搜索一次最佳阈值（更公平）
            f1w, tau_pack, pred = search_best_hier_thresholds(ys, probs_ens)

        cm = confusion_matrix(ys, pred, labels=[0, 1, 2])
        cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
        print("\n==== [Ensemble] ====")
        print(f"阈值: {tau_pack} | F1(w)={f1w:.4f}")
        print("混淆矩阵（行=真值 / 列=预测）:\n", cm)
        print("行归一化：\n", np.round(cmn, 2))
        per = f1_score(ys, pred, average=None, labels=[0, 1, 2], zero_division=0)
        for i, n in enumerate(EVENT_CLASS_NAMES):
            print(f"  {n}: F1={per[i]:.4f}")

        results.append({
            "name": "Ensemble",
            "f1_weighted": float(f1w),
            "tau": tau_pack,
            "cm": cm.tolist(),
            "cm_norm": cmn.tolist(),
        })

    # 保存 CSV（简要）
    if results:
        import csv
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "F1_weighted", "tau_abn", "tau_h", "tau_oa"])
            for r in results:
                t = r["tau"]
                w.writerow([r["name"], f"{r['f1_weighted']:.6f}", t["tau_abn"], t["tau_h"], t["tau_oa"]])
        print(f"[Saved] {out_csv.resolve()}")


if __name__ == "__main__":
    main()
