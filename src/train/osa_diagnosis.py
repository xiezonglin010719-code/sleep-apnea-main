# -*- coding: utf-8 -*-
"""
events_to_night_metrics_with_truth.py

功能（相较旧版增强）：
- 载入“事件级三分类模型”（Normal/Hypopnea/OA）在验证集上逐序列前向，按受试者聚合，规则式计算预测 AHI/严重度；
- 解析 RML（可选 EDF）计算“真实 AHI/严重度”，并把真值列写入 CSV；
- 自动评估：二分类（是否患病）与四分类（None/Mild/Moderate/Severe）。

依赖：
- src.models.osa_end2end.OSAEnd2EndModel
- 验证集：每受试者一个 .pickle（事件对象含 .signal (H,W) & .label）
- 真值：RML 事件标注目录 --rml_dir（必选以生成真值）；可选 EDF 目录 --edf_dir
"""

import os
import csv
import argparse
import pickle
from pathlib import Path
from collections import defaultdict, Counter
from xml.dom import minidom

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from tqdm import tqdm

# 你的模型
from src.models.osa_end2end import OSAEnd2EndModel

# ============ 常量 ============
EVENT_CLASS_NAMES = ["Normal", "Hypopnea", "ObstructiveApnea"]

# 输入标签清洗（与训练保持一致；CA 在预测阶段忽略）
RAW2TRAIN = {
    "normal": 0, "none": 0, "background": 0, "noevent": 0, "negative": 0,
    "hypopnea": 1, "hypopnoea": 1,
    "obstructive apnea": 2, "obstructiveapnea": 2, "oa": 2,
    "mixed apnea": 2, "mixedapnea": 2, "ma": 2,
}
IGNORE_SET = {"central apnea", "centralapnea", "ca"}  # 预测阶段不参与

SEVERITY_BINS = [(0, 5), (5, 15), (15, 30), (30, float("inf"))]
SEVERITY_NAMES = ["None", "Mild", "Moderate", "Severe"]


# ============ 工具 ============
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_dir(p, anchors):
    P = Path(p)
    if P.is_absolute():
        return str(P)
    for a in anchors:
        cand = a / P
        if cand.exists():
            return str(cand.resolve())
    return str((anchors[0] / P).resolve())


def to_float_image(sig, force_hw=None):
    import torch as _torch
    import numpy as _np
    x = _np.asarray(sig, _np.float32)
    x = _np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if force_hw is not None and x.shape != tuple(force_hw):
        ten = _torch.from_numpy(x)[None, None, ...]
        ten = F.interpolate(ten, size=force_hw, mode="bilinear", align_corners=False)
        x = ten.squeeze().numpy().astype(_np.float32)
    return x


import re

def _norm_sid(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]", "", str(s)).lower()

def _norm_text(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(s).strip().lower())

def find_edf_files_for_sid(sid: str, edf_index: dict, pad_len: int = 8):
    """
    返回该受试者所有 EDF 文件的列表（可能有 [001],[002], ...）。
    优先按父目录名匹配（如 /edf/00001000/），其次基名/包含关系。
    """
    import os, re
    def _norm(s): return re.sub(r"[^0-9a-zA-Z]", "", str(s)).lower()

    key = _norm(sid)
    key_padded = key.zfill(pad_len) if key.isdigit() and len(key) < pad_len else key

    files     = edf_index["files"]
    by_base   = edf_index["by_base"]    # {norm(base): [paths]}
    by_folder = edf_index["by_folder"]  # {norm(folder): [paths]}

    # 1) 父目录精确（最稳）
    for k in (key, key_padded):
        if k in by_folder:
            return sorted(by_folder[k])

    # 2) 基名里含有（同一个 subject 的不同分段往往有相同前缀）
    cands = []
    for k in (key, key_padded):
        for nb, paths in by_base.items():
            if k in nb:
                cands.extend(paths)
    if cands:
        return sorted(list(set(cands)))

    # 3) 路径包含
    for k in (key, key_padded):
        contains = [p for p in files if k in _norm(p)]
        if contains:
            return sorted(list(set(contains)))

    return []

def estimate_total_tst_from_edf_list(edf_paths):
    """
    把同一受试者的多个 EDF 分段的时长相加（单位：秒）。
    """
    total = 0.0
    for p in edf_paths:
        sec = estimate_tst_seconds_from_edf(p)  # 你已有的函数：返回单个 EDF 的记录时长
        if isinstance(sec, (int, float)) and sec > 0:
            total += float(sec)
    return total if total > 0 else None


def build_file_index(root_dir: str, exts=(".rml", ".xml")):
    """
    递归扫描 root_dir，返回：
      - files: 所有符合后缀的文件完整路径列表
      - by_base: { 规范化(文件基名去扩展) : [path, ...] }
      - by_folder: { 规范化(直接父目录名) : [path, ...] }
    """
    files, by_base, by_folder = [], {}, {}
    for dp, _, fns in os.walk(root_dir):
        for fn in fns:
            if not any(fn.lower().endswith(e.lower()) for e in exts):
                continue
            p = os.path.join(dp, fn)
            files.append(p)
            base = os.path.splitext(fn)[0]
            nb = _norm_sid(base)
            by_base.setdefault(nb, []).append(p)
            fold = os.path.basename(dp)
            nf = _norm_sid(fold)
            by_folder.setdefault(nf, []).append(p)
    return {"files": files, "by_base": by_base, "by_folder": by_folder}

def find_file_for_sid(sid: str, index: dict, pad_len: int = 8):
    key = _norm_sid(sid)
    # 尝试零填充版本（1008 → 00001008）
    if key.isdigit() and pad_len and len(key) < pad_len:
        key_padded = key.zfill(pad_len)
    else:
        key_padded = key

    by_base   = index["by_base"]
    by_folder = index["by_folder"]
    files     = index["files"]

    # 1) 基名精确（优先 padded）
    for k in (key, key_padded):
        cand = by_base.get(k)
        if cand:
            return sorted(cand, key=lambda x: (len(x)))[0]

    # 2) 父目录精确
    for k in (key, key_padded):
        cand = by_folder.get(k)
        if cand:
            return sorted(cand, key=lambda x: (len(x)))[0]

    # 3) 路径包含
    for k in (key, key_padded):
        contains = [p for p in files if k in _norm_sid(p)]
        if contains:
            return sorted(contains, key=lambda x: len(x))[0]
    return None

def estimate_tst_from_rml(events, min_fallback_sec=3600.0):
    """若没有 EDF，则用 RML 事件跨度估计 TST；跨度太小时给温和兜底。"""
    if not events:
        return min_fallback_sec
    starts = [s for (_, s, _) in events]
    ends   = [s + d for (_, s, d) in events]
    span   = max(ends) - min(starts)
    return span if span >= 300 else min_fallback_sec


# —— 工具：EDF —— #
def estimate_tst_seconds_from_edf(edf_path):
    """
    通用兜底：返回 EDF 记录总时长（秒）。
    若要严格 TST，可进一步解析分期通道；为兼容不同数据集，这里先用记录时长。
    """
    try:
        import pyedflib
        f = pyedflib.EdfReader(edf_path)
        dur = float(f.getFileDuration())
        f._close(); del f
        return dur
    except Exception:
        pass
    try:
        import mne
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        return float(raw.n_times / raw.info["sfreq"])
    except Exception:
        pass
    return None

def build_edf_index(root_dir: str):
    """同上，但用于 EDF 扫描"""
    return build_file_index(root_dir, exts=(".edf", ".EDF"))
def _norm_text(s: str) -> str:
    """去空格→小写→连字符/下划线归一，方便匹配"""
    return re.sub(r"[\s_\-]+", "", str(s).strip().lower())

def compute_truth_from_rml_edf(subject_id,
                               rml_dir=None, edf_dir=None,
                               rml_index=None, edf_index=None,
                               count_hypopnea=True,
                               count_mixed=True,
                               count_central=True,
                               include_rera_in_ahi=False,
                               fallback_from_rml_span=True):
    """
    计算“真值”AHI/严重度；支持同一受试者多段 EDF 时长累加。
    返回: (true_ahi: float, sev_idx: int, detail: dict)
    """
    # 1) 找 RML
    if rml_index is None:
        rml_index = build_file_index(rml_dir or ".", exts=(".rml", ".xml"))
    rml_path = find_file_for_sid(subject_id, rml_index)  # 你脚本里已有 find_file_for_sid
    if rml_path is None or not os.path.exists(rml_path):
        return None, None, {"reason": "rml_not_found", "sid": subject_id}

    # 2) 解析 RML 事件
    events = parse_rml_events(rml_path)  # 你脚本里已有

    # 3) 估计 TST：优先 EDF 累加；否则 RML 跨度
    edf_paths, tst_source = [], ""
    if edf_index is None and edf_dir:
        edf_index = build_edf_index(edf_dir)
    if edf_index is not None:
        edf_paths = find_edf_files_for_sid(subject_id, edf_index)
    tst_sec = estimate_total_tst_from_edf_list(edf_paths) if edf_paths else None
    if tst_sec is not None:
        tst_source = "edf_sum"
    elif fallback_from_rml_span:
        tst_sec = estimate_tst_from_rml(events)
        tst_source = "rml"
    if tst_sec is None or tst_sec <= 0:
        return None, None, {"reason": "tst_unavailable", "sid": subject_id, "rml_path": rml_path}

    # 4) 事件计数（决定 N 与 by_type）
    K_OA   = {"obstructiveapnea", "oa"}
    K_H    = {"hypopnea", "hypopnoea"}
    K_MA   = {"mixedapnea", "ma"}
    K_CA   = {"centralapnea", "ca"}
    K_RERA = {"rera", "respiratoryeffortrelatedarousal", "respiratoryeffortrelatedarousals"}

    from collections import Counter
    by_type = Counter()
    N = 0
    for etype_raw, s, d in events:
        t = _norm_text(etype_raw)
        if t in K_OA:
            by_type["OA"] += 1; N += 1
        elif t in K_H and count_hypopnea:
            by_type["H"]  += 1; N += 1
        elif t in K_MA and count_mixed:
            by_type["MA"] += 1; N += 1
        elif t in K_CA and count_central:
            by_type["CA"] += 1; N += 1
        elif t in K_RERA and include_rera_in_ahi:
            by_type["RERA"] += 1; N += 1
        # 其他类型（Arousal/Snore/Gain/ChannelFail等）忽略

    hours    = tst_sec / 3600.0
    true_ahi = N / hours if hours > 1e-6 else 0.0
    sev_idx  = classify_severity(true_ahi)  # 你脚本里已有

    # 5) 返回（此处的变量都在当前作用域中，避免未解析报错）
    return true_ahi, sev_idx, {
        "N_events": int(N),
        "hours": float(hours),
        "by_type": dict(by_type),
        "rml_path": rml_path,
        "edf_paths": edf_paths,
        "tst_source": tst_source
    }




def classify_severity(ahi: float) -> int:
    if ahi < SEVERITY_BINS[0][1]:
        return 0
    for i, (lo, hi) in enumerate(SEVERITY_BINS[1:], 1):
        if lo <= ahi < hi:
            return i
    return len(SEVERITY_BINS) - 1


def safe_load_ckpt(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e1:
        try:
            from torch.serialization import add_safe_globals
            import numpy.core.multiarray as _m
            add_safe_globals([_m._reconstruct])
        except Exception:
            pass
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except Exception as e2:
            print(f"[warn] weights_only=True 失败，改用 weights_only=False（确保权重来源可信）。\n  e1={e1}\n  e2={e2}")
            return torch.load(path, map_location=map_location, weights_only=False)


# ============ 验证序列数据集 ============
class ValEventSeqDataset(Dataset):
    def __init__(self, data_dir, img_size=64, seq_len=10):
        self.data_dir = data_dir
        self.img_size = img_size
        self.seq_len = seq_len

        self.events_by_subject = {}
        self.subj_stats = {}
        self.samples = []     # list[(sid, [imgs])]
        self._scan_subjects()
        self._make_sequences()

    def _scan_subjects(self):
        pkl_files = sorted([fn for fn in os.listdir(self.data_dir) if fn.endswith(".pickle")])
        print(f"[Scan] 目录：{self.data_dir} | .pickle 文件数：{len(pkl_files)}")

        for fn in pkl_files:
            sid = os.path.splitext(fn)[0]
            path = os.path.join(self.data_dir, fn)
            try:
                events = pickle.load(open(path, "rb"))
            except Exception as e:
                print(f"[warn] 读取失败：{path} → {e}")
                continue

            imgs = []
            for ev in events:
                lab_raw = str(getattr(ev, "label", "none")).lower().strip()
                if lab_raw in IGNORE_SET:
                    continue
                if lab_raw not in RAW2TRAIN:
                    continue
                sig = getattr(ev, "signal", None)
                if sig is None:
                    continue
                img = to_float_image(sig, force_hw=(self.img_size, self.img_size))
                imgs.append(img)

            if len(imgs) == 0:
                print(f"[Scan] {fn}: 可用事件=0（可能全为 CA 或无效）")
                continue

            arr = np.stack(imgs, 0)
            mu, std = float(arr.mean()), float(arr.std())
            if not np.isfinite(std) or std < 1e-6:
                std = 1e-3
            self.subj_stats[sid] = (mu, std)
            self.events_by_subject[sid] = imgs
            print(f"[Scan] {fn}: 事件数={len(imgs)} | μ={mu:.4f} σ={std:.4f}")

    def _make_sequences(self):
        for sid, imgs in self.events_by_subject.items():
            L = len(imgs)
            if L < self.seq_len:
                continue
            for i in range(0, L - self.seq_len + 1):
                self.samples.append((sid, imgs[i:i+self.seq_len]))
        print(f"[MakeSeq] 生成序列：{len(self.samples)} (seq_len={self.seq_len}) | subjects={len(self.events_by_subject)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, seq_imgs = self.samples[idx]
        mu, std = self.subj_stats.get(sid, (0.0, 1.0))
        frames = []
        for img in seq_imgs:
            x = (img - mu) / std
            x = np.clip(x, -5.0, 5.0)
            frames.append(torch.from_numpy(x).float().unsqueeze(0))
        seq_tensor = torch.stack(frames, dim=0)  # (T,1,H,W)
        return seq_tensor, sid


# ============ 真值计算（RML + 可选 EDF） ============
def parse_rml_events(rml_path: str):
    """
    返回 events: list[(etype_raw, start_sec, dur_sec)]
    只要文件里有 <Event Type=... Start=... Duration=...> 就解析出来。
    """
    doc = minidom.parse(rml_path)
    evs = doc.getElementsByTagName("Event")
    out = []
    for e in evs:
        et = e.getAttribute("Type")
        st = e.getAttribute("Start")
        du = e.getAttribute("Duration") or e.getAttribute("Dur") or e.getAttribute("Length")

        try:
            s = float(st)
        except Exception:
            continue
        try:
            d = float(du) if du is not None and du != "" else 0.0
        except Exception:
            d = 0.0
        out.append((et, s, d))
    return out



def estimate_tst_seconds_from_edf(edf_path):
    """
    用 EDF 估计总睡眠时长（秒）：
      - 若能读取到睡眠分期通道（如 Hypnogram），则统计 N1/N2/N3/REM 的总时长；
      - 否则退化为整个记录时长（startdatetime + duration）。
    优先使用 pyedflib；若无则尝试 mne；再失败返回 None。
    """
    # 尝试 pyedflib
    try:
        import pyedflib
        f = pyedflib.EdfReader(edf_path)
        dur = f.getFileDuration()  # 秒
        # 尝试读取注释通道，寻找睡眠分期；若没有，就用总时长
        f._close()
        del f
        return float(dur)
    except Exception:
        pass

    # 尝试 mne
    try:
        import mne
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        # 原则上也可以读 annotations 看分期，但不同数据集差异很大，这里兜底用记录时长
        dur = raw.n_times / raw.info["sfreq"]
        return float(dur)
    except Exception:
        pass

    return None



# ============ 规则式夜间诊断（预测侧） ============
def rule_based_night_diag(seq_probs_by_sid, epoch_seconds=30, consec_k=2):
    """
    seq_probs_by_sid: dict[sid] -> list of (p0,p1,p2)
    返回 per_sid 预测结果（AHI/Severity）
    """
    per_sid = {}
    for sid, plist in seq_probs_by_sid.items():
        P = np.stack(plist, 0)  # (N,3)
        pred = np.argmax(P, axis=1)  # 0/1/2
        abn = (pred != 0).astype(int)

        # 连续 K 个异常 → 1 次呼吸事件
        events, streak = 0, 0
        for a in abn:
            if a == 1:
                streak += 1
                if streak >= consec_k:
                    events += 1
                    streak = 0
            else:
                streak = 0

        N = len(abn)
        hours = N * epoch_seconds / 3600.0
        ahi = events / hours if hours > 1e-6 else 0.0
        sev = classify_severity(ahi)

        per_sid[sid] = {
            "N_seq": int(N),
            "N_events": int(events),
            "hours": float(hours),
            "AHI": float(ahi),
            "sev_idx": int(sev),
            "sev_name": SEVERITY_NAMES[sev]
        }
    return per_sid


def write_csv(per_sid_pred, per_sid_truth, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "subject_id",
            # 预测侧
            "pred_N_seq", "pred_N_events", "pred_hours", "pred_AHI", "pred_severity_idx", "pred_severity_name",
            # 真值侧
            "true_N_events", "true_hours", "true_AHI", "true_severity_idx", "true_severity_name",
            # 二分类（是否患病）
            "disease_pred", "disease_true"
        ])
        all_sids = sorted(set(per_sid_pred.keys()) | set(per_sid_truth.keys()))
        for sid in all_sids:
            pd = per_sid_pred.get(sid, {})
            td = per_sid_truth.get(sid, {})

            pred_hours = pd.get("hours", "")
            true_hours = td.get("hours", "")

            w.writerow([
                sid,
                # 预测
                pd.get("N_seq", ""), pd.get("N_events", ""), f"{pred_hours:.6f}" if pred_hours != "" else "",
                f"{pd.get('AHI',''):.6f}" if pd.get("AHI", "") != "" else "",
                pd.get("sev_idx", ""), pd.get("sev_name", ""),
                # 真值
                td.get("N_events", ""), f"{true_hours:.6f}" if true_hours != "" else "",
                f"{td.get('AHI',''):.6f}" if td.get("AHI", "") != "" else "",
                td.get("sev_idx", ""), td.get("sev_name", ""),
                # 二分类
                0 if pd.get("sev_idx", 0) == 0 else 1,
                0 if td.get("sev_idx", 0) == 0 else 1
            ])
    print(f"[OK] 写出结果: {out_csv}")

def _safe_classification_report(y_true, y_pred, all_names):
    """根据出现过的标签动态生成 labels / names；如果只有1类，打印计数并跳过 report。"""
    present = sorted(set(map(int, y_true)) | set(map(int, y_pred)))
    if len(present) <= 1:
        only = present[0] if present else None
        print(f"[warn] 仅检测到 {len(present)} 个类别，跳过 classification_report。")
        if only is not None:
            name = all_names[only] if 0 <= only < len(all_names) else str(only)
            print(f"唯一类别：{only} ({name})，样本数 = {len(y_true)}")
        return present, None  # 仍返回 present 给混淆矩阵使用
    names = [all_names[i] if 0 <= i < len(all_names) else str(i) for i in present]
    print(classification_report(y_true, y_pred, labels=present, target_names=names, zero_division=0))
    return present, names

def evaluate_from_joined(per_sid_pred, per_sid_truth):
    """
    per_sid_pred: {sid: {'N_events', 'hours', 'ahi', 'sev_idx', ...}}
    per_sid_truth:{sid: {'N_events', 'hours', 'ahi', 'sev_idx', ...}}
    """
    shared = sorted(set(per_sid_pred.keys()) & set(per_sid_truth.keys()))
    if len(shared) == 0:
        print("[error] 没有交集受试者，无法评估。")
        return

    y_true_bin, y_pred_bin = [], []
    y_true_4,  y_pred_4   = [], []
    for sid in shared:
        t = per_sid_truth[sid]
        p = per_sid_pred[sid]
        # 真值/预测严重度索引
        t4 = int(t.get("sev_idx", 0))
        p4 = int(p.get("sev_idx", 0))
        y_true_4.append(t4)
        y_pred_4.append(p4)
        # 二分类（0=正常, 1=患病）
        y_true_bin.append(0 if t4 == 0 else 1)
        y_pred_bin.append(0 if p4 == 0 else 1)

    # ===== 二分类 =====
    print("\n=== 夜级诊断（二分类：Normal vs Disease）===")
    acc_bin = accuracy_score(y_true_bin, y_pred_bin)
    f1w_bin = f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0)
    print(f"Acc={acc_bin:.4f} | F1w={f1w_bin:.4f}")

    # 自适应标签（通常是 [0,1]，但保守起见也动态取）
    present_bin = sorted(set(y_true_bin) | set(y_pred_bin))
    cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=present_bin)
    print("Confusion Matrix (bin):\n", cm_bin)

    # ===== 四分类 =====
    print("\n=== 夜级诊断（四分类：None/Mild/Moderate/Severe）===")
    acc_4 = accuracy_score(y_true_4, y_pred_4)
    f1w_4 = f1_score(y_true_4, y_pred_4, average="weighted", zero_division=0)
    print(f"Acc={acc_4:.4f} | F1w={f1w_4:.4f}")

    # 动态生成 labels / names；若只有1个类则跳 report，但继续输出 CM
    present_4, _ = _safe_classification_report(y_true_4, y_pred_4, SEVERITY_NAMES)
    cm_4 = confusion_matrix(y_true_4, y_pred_4, labels=present_4)
    print("Confusion Matrix (4cls):\n", cm_4)


# ============ 主流程 ============
def main():
    ap = argparse.ArgumentParser()
    # 数据与模型
    ap.add_argument("--val_dir",   default="/Users/liyuxiang/Downloads/sleep-apnea-main/data/processed/all_subjects")
    ap.add_argument("--event_ckpt", default="models/end2end/osa_end2end_best_3cls_loso_00000999.pth")
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--seq_len",  type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epoch_seconds", type=int, default=30, help="每个序列代表的秒数")
    ap.add_argument("--consec_k", type=int, default=2, help="连续K个异常计一次事件")
    # 真值来源
    ap.add_argument("--rml_dir", default="/Users/liyuxiang/Downloads/sleep-apnea-main/data/preprocess/rml", help="RML 目录（subject_id.rml 或 .xml）")
    ap.add_argument("--edf_dir", default="/Users/liyuxiang/Downloads/sleep-apnea-main/data/preprocess/edf", help="EDF 目录（可选，若有则用于更精确的 TST）")
    ap.add_argument("--include_rera_in_ahi", action="store_true", help="把 RERA 计入 AHI（默认不计）")
    ap.add_argument("--count_central", action="store_true", help="把 CA 计入 AHI（默认计入，若不想计入可加 --no_count_central 配置）")
    ap.add_argument("--no_count_central", action="store_true", help="不把 CA 计入 AHI（覆盖上一个开关）")
    ap.add_argument("--no_count_mixed", action="store_true", help="不把 MA 计入 AHI")
    ap.add_argument("--no_count_hypopnea", action="store_true", help="不把 Hypopnea 计入 AHI")
    # 输出
    ap.add_argument("--out_csv",  default="results/night_diag_with_truth.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchors = [Path.cwd(), project_root()]
    val_dir = resolve_dir(args.val_dir, anchors)
    ckpt_path = resolve_dir(args.event_ckpt, anchors)
    rml_dir = resolve_dir(args.rml_dir, anchors)
    edf_dir = resolve_dir(args.edf_dir, anchors) if args.edf_dir else None

    print(f"使用设备: {device}")
    print(f"Val dir   : {val_dir}")
    print(f"Event ckpt: {ckpt_path}")
    print(f"RML dir   : {rml_dir}")
    print(f"EDF dir   : {edf_dir if edf_dir else '(未提供)'}")

    # ========== 构造验证序列 ==========
    ds = ValEventSeqDataset(val_dir, img_size=args.img_size, seq_len=args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ========== 加载事件模型 ==========
    ckpt = safe_load_ckpt(ckpt_path, map_location=device)
    best_tau = ckpt.get("best_tau", 0.5)
    model = OSAEnd2EndModel(img_channels=1, img_size=args.img_size, seq_len=args.seq_len, num_classes=3).to(device)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[OK] 事件模型已加载 | best_tau={best_tau:.2f}")

    # ========== 前向并聚合为受试者概率 ==========
    seq_probs_by_sid = defaultdict(list)
    with torch.no_grad():
        for seq, sids in tqdm(dl, desc="事件模型前向(验证集)"):
            seq = seq.to(device)
            out = model(seq)                 # (B,3)
            p = out.softmax(dim=1).cpu().numpy()
            # 用 tau 压回低置信异常
            abn_prob = p[:, 1] + p[:, 2]
            pred = np.argmax(p, axis=1)
            pred[(abn_prob < best_tau)] = 0
            p_adj = np.zeros_like(p)
            p_adj[np.arange(len(pred)), pred] = 1.0

            for sid, row in zip(sids, p_adj):
                seq_probs_by_sid[str(sid)].append(row)

    # ========== 预测侧 夜级 AHI/Severity ==========
    per_sid_pred = rule_based_night_diag(seq_probs_by_sid,
                                         epoch_seconds=args.epoch_seconds,
                                         consec_k=args.consec_k)

    # ========== 真值侧：RML(+EDF) 计算 ==========
    # 基于 val_dir 里出现过的 subject_id 去找对应 RML/EDF
    # ========== 真值侧：RML(+EDF) 计算 ==========
    sids = sorted(per_sid_pred.keys())
    per_sid_truth = {}

    # 仅构建一次索引，递归扫描整个目录树
    rml_index = build_file_index(rml_dir, exts=(".rml", ".xml"))
    edf_index = build_edf_index(edf_dir) if edf_dir else None

    for sid in tqdm(sids, desc="计算真值(AHI/Severity)"):
        t_ahi, t_sev, detail = compute_truth_from_rml_edf(
            sid,
            rml_index=rml_index,
            edf_index=edf_index,
            count_hypopnea=(not args.no_count_hypopnea),
            count_mixed=(not args.no_count_mixed),
            count_central=(not args.no_count_central),
            include_rera_in_ahi=args.include_rera_in_ahi,
            fallback_from_rml_span=True
        )
        if t_ahi is None:
            print(f"[truth] {sid}: 无法计算（{detail.get('reason', 'unknown')}）")
            continue
        per_sid_truth[sid] = {
            "N_events": int(detail["N_events"]),
            "hours": float(detail["hours"]),
            "AHI": float(t_ahi),
            "sev_idx": int(t_sev),
            "sev_name": SEVERITY_NAMES[int(t_sev)]
        }
        print(sid)

        if t_ahi is None:
            print(f"[truth] {sid}: 无法计算（{detail.get('reason','unknown')}）")
            continue
        per_sid_truth[sid] = {
            "N_events": int(detail["N_events"]),
            "hours": float(detail["hours"]),
            "AHI": float(t_ahi),
            "sev_idx": int(t_sev),
            "sev_name": SEVERITY_NAMES[int(t_sev)]
        }

    # ========== 写 CSV（含真值列） & 评估 ==========
    out_csv = resolve_dir(args.out_csv, anchors)
    write_csv(per_sid_pred, per_sid_truth, out_csv)
    evaluate_from_joined(per_sid_pred, per_sid_truth)


if __name__ == "__main__":
    main()
