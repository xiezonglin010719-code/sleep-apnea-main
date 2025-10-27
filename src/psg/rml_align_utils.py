# -*- coding: utf-8 -*-
from typing import List, Dict, Tuple
import numpy as np

def _norm_name(name: str) -> str:
    n = (name or "").strip().lower()
    n = n.replace(" ", "").replace("_", "")
    n = n.replace("apnoea", "apnea")
    if n.endswith("event"):
        n = n[:-5]
    return n

def build_label_mapping(cfg_event_labels: dict) -> Dict[str, int]:
    # 例：{"Normal":0,"ObstructiveApnea":1,"CentralApnea":2,"MixedApnea":3,"Hypopnea":4}
    mapping = {}
    for k, v in (cfg_event_labels or {}).items():
        mapping[_norm_name(k)] = int(v)
    # 常见别名
    aliases = {
        "obstructiveapnea": ["obstructive", "oa"],
        "centralapnea": ["central", "ca"],
        "mixedapnea": ["mixed", "ma"],
        "hypopnea": ["hypopnoea", "hyp"],
        "normal": ["none", "baseline", "noevent"],
    }
    for std, alist in aliases.items():
        if std in mapping:
            for a in alist:
                mapping[a] = mapping[std]
    return mapping

def align_with_segments_robust(
    events: List[Dict],               # [{'start':sec,'end':sec,'label':str}, ...]
    segment_length: float,            # 30.0
    segment_overlap: float,           # 0.5
    total_duration: float,            # 3600.0
    label_mapping: Dict[str, int],    # 来自 build_label_mapping
    overlap_threshold_sec: float = 3.0,
    keep_only_resp: bool = True
) -> List[int]:
    """
    用“重叠阈值 + 事件优先级”把 RML 事件对齐到每个 30s 段。
    - 默认只保留呼吸相关（Apnea/Hypopnea），忽略 Arousal/LegMovement 等。
    - 段内多事件时按优先级选（Obstructive > Mixed > Central > Hypopnea > Normal）。
    """
    # 定义呼吸类事件集合（规范化后）
    RESP = {"obstructiveapnea", "hypopnea"}
    # 优先级（数字越小优先级越高）可按需求调整
    PRIO = {"obstructiveapnea": 1, "hypopnea": 4, "normal": 9}

    # 规范化事件 & 可选过滤
    evs = []
    for ev in events:
        try:
            s = float(ev.get("start", 0.0))
            e = float(ev.get("end", 0.0))
        except Exception:
            # 有些 RML 是 start+duration
            dur = float(ev.get("duration", 0.0))
            s = float(ev.get("start", 0.0))
            e = s + dur
        raw = str(ev.get("label", "") or ev.get("type", "") or ev.get("concept", ""))
        lab = _norm_name(raw)
        if keep_only_resp and (lab not in RESP):
            continue
        if lab not in label_mapping:
            # 未配置的标签忽略（避免全归 0）
            continue
        evs.append((max(0.0, s), max(0.0, e), lab, label_mapping[lab]))

    # 段切分
    step = segment_length * (1.0 - segment_overlap)
    n_seg = int((total_duration - segment_length) // max(step, 1e-6)) + 1
    labels = []
    normal_id = label_mapping.get("normal", 0)

    for i in range(n_seg):
        seg_s = i * step
        seg_e = seg_s + segment_length
        cand = []
        for (es, ee, lab, lab_id) in evs:
            ov = max(0.0, min(seg_e, ee) - max(seg_s, es))
            if ov >= overlap_threshold_sec:
                cand.append((PRIO.get(lab, 99), lab_id))
        if not cand:
            labels.append(normal_id)
        else:
            cand.sort(key=lambda x: x[0])
            labels.append(cand[0][1])
    return labels
