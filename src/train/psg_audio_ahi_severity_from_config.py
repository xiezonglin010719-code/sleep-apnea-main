#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成 PSG-Audio 的夜级真值标签（AHI & 严重度）——读取 YAML 配置驱动
- 可选 --rml-dir 指定 RML 根目录（优先级最高），否则依据 config.paths.root 搜索
- 递归扫描 *.rml，支持从文件名或父目录提取 subject_id
- AHI 计数：Obstructive + Hypopnea (+ 可选 Central/Mixed)
- 严重度阈值：<5 / [5,15) / [15,30) / ≥30 → None/Mild/Moderate/Severe
- TST：优先累计睡眠分期(N1/N2/N3/REM)，否则退回记录覆盖时长
"""

import os
import re
import csv
import json
import argparse
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    import yaml
except Exception as e:
    raise SystemExit("缺少依赖：pyyaml，请先 `pip install pyyaml`") from e


# ====================== 常量定义 ======================
SEVERITY_BINS = [(0, 5), (5, 15), (15, 30), (30, float("inf"))]
SEVERITY_NAMES = ["None", "Mild", "Moderate", "Severe"]

RESP_BASE = {"obstructive apnea", "hypopnea"}
RESP_EXTRA = {"central apnea", "mixed apnea"}  # 可选计入

SLEEP_STAGE_POSITIVE = {
    "n1", "n2", "n3", "rem", "stage n1", "stage n2", "stage n3", "stage rem", "sleep"
}
SLEEP_STAGE_WAKE = {"w", "wake", "stage w"}


# ====================== 小工具函数 ======================
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

from collections import Counter

def debug_label_hist(events, topk=40):
    c = Counter()
    for e in events:
        c[e.label.strip()] += 1
    print("[DEBUG] Top labels:")
    for lab, cnt in c.most_common(topk):
        print(f"  {cnt:5d}  {lab}")


def resolve_config_path(cli_path: Optional[str]) -> str:
    """给 --config 一个稳健的解析（默认项目根 config.yaml or 脚本同目录）"""
    if cli_path and os.path.isfile(cli_path):
        return cli_path
    # 常见候选：脚本同目录、脚本上两级
    here = Path(__file__).resolve().parent
    cands = [
        here / "config.yaml",
        here.parent / "config.yaml",
        here.parent.parent / "config.yaml",
    ]
    for p in cands:
        if p.is_file():
            return str(p)
    raise SystemExit(
        "找不到配置文件。请使用 --config 指定，或将 config.yaml 放在：\n"
        + "\n".join([str(x) for x in cands])
    )


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def localname(tag: str) -> str:
    """去命名空间 -> 本地名小写"""
    if not tag:
        return ""
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    return tag.strip().lower()


def child_text_any(node: ET.Element, cand_names) -> str:
    """在候选标签名集合里找第一个存在的子节点并取文本（大小写/命名空间不敏感）"""
    lut = {}
    for ch in list(node):
        lut.setdefault(localname(ch.tag), []).append(ch)
    for n in cand_names:
        key = n.strip().lower()
        if key in lut and lut[key]:
            txt = (lut[key][0].text or "").strip()
            if txt != "":
                return txt
    return ""


def parse_time_to_seconds(s: str) -> Optional[float]:
    """将 '123.4' / '00:12:34.5' / '12:34' / '12345 ms' 解析为秒"""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    m_ms = re.match(r"^\s*(\d+(?:\.\d+)?)\s*ms\s*$", s, re.I)
    if m_ms:
        return float(m_ms.group(1)) / 1000.0
    if re.match(r"^\d+(?:\.\d+)?$", s):  # 纯数字：秒
        return float(s)
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
    except Exception:
        pass
    return None


# ====================== 事件结构 & 解析 ======================
class Event:
    def __init__(self, label: str, start: float, duration: float):
        self.label = label
        self.start = float(start)
        self.duration = float(max(0.0, duration))

    @property
    def end(self) -> float:
        return self.start + self.duration


def parse_rml_file(path: str) -> List[Event]:
    out: List[Event] = []
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as e:
        print(f"[WARN] 解析失败，跳过: {path} ({e})")
        return out

    evt_nodes = []
    for n in root.iter():
        if localname(n.tag) in ("scoredevent", "event"):
            evt_nodes.append(n)

    for n in evt_nodes:
        # 仍旧尝试“浅层字段”，方便调试
        concept = child_text_any(n, ["EventConcept", "Concept", "Name", "EventName"])
        etype   = child_text_any(n, ["EventType", "EventFamily", "Type", "Family"])
        subtype = child_text_any(n, ["EventSubType", "SubType"])
        label = " ".join([x for x in [etype, concept, subtype] if x]).strip()

        # 生成“深度文本池”（包含所有后代与属性）
        full_text = collect_all_texts(n)

        # 时间
        start_txt = child_text_any(n, ["Start", "StartSec", "StartTime", "Onset", "Begin", "StartTimeSec"])
        dur_txt   = child_text_any(n, ["Duration", "DurationSec", "Length", "DurationMS", "Dur", "Len"])
        end_txt   = child_text_any(n, ["End", "EndSec", "Stop", "Offset"])

        if not start_txt:
            for k, v in (n.attrib or {}).items():
                lk = localname(k)
                if lk in ("start","startsec","starttime","onset","begin"):
                    start_txt = v.strip()
                elif lk in ("duration","durationsec","length","durationms","dur","len"):
                    dur_txt = dur_txt or v.strip()
                elif lk in ("end","endsec","stop","offset"):
                    end_txt = end_txt or v.strip()

        start = parse_time_to_seconds(start_txt) if start_txt else None
        duration = parse_time_to_seconds(dur_txt) if dur_txt else None
        end = parse_time_to_seconds(end_txt) if end_txt else None

        if start is not None and duration is None and end is not None and end >= start:
            duration = end - start

        if start is None or duration is None:
            continue

        # ✅ 修复这里
        e = Event(label=(label if label else "Unknown"), start=start, duration=duration)
        e.full_text = full_text.lower()
        out.append(e)

    return out


# ====================== AHI/TST 计算 ======================
def compute_tst_seconds_from_events(events: List[Event]) -> Optional[float]:
    """优先用睡眠分期累计 TST，否则退回到记录覆盖时长"""
    if not events:
        return None
    pos_sum = 0.0
    found_stage = False
    for e in events:
        lbl = norm(e.label)
        if lbl in SLEEP_STAGE_POSITIVE:
            pos_sum += e.duration
            found_stage = True
        elif lbl in SLEEP_STAGE_WAKE:
            found_stage = True
    if found_stage and pos_sum > 0:
        return pos_sum

    starts = [e.start for e in events]
    ends = [e.end for e in events]
    if not starts or not ends:
        return None
    return max(ends) - min(starts)


RE_APNEA  = re.compile(r"\bapne[ao]\b", re.I)          # apnea/apnoea
RE_HYPO   = re.compile(r"hypopn?o?e?a?", re.I)         # hypopnea
RE_OBSTR  = re.compile(r"obstru", re.I)                # obstructive
RE_CENT   = re.compile(r"central", re.I)
RE_MIXED  = re.compile(r"mixed", re.I)
RE_OA     = re.compile(r"\bOA\b", re.I)
RE_CA     = re.compile(r"\bCA\b", re.I)
RE_MA     = re.compile(r"\bMA\b", re.I)

def count_respiratory_events(events: List[Event], include_central_mixed: bool = True) -> Tuple[int, Dict[str, int]]:
    total = 0
    counts = {"hypopnea": 0, "obstructive apnea": 0}
    if include_central_mixed:
        counts.update({"central apnea": 0, "mixed apnea": 0})

    for e in events:
        text = getattr(e, "full_text", (e.label or "")).lower()

        # 必须是呼吸相关：有些数据把大类写成 Respiratory，再把细分写在别处
        if "respiratory" not in text:
            # 有的不会出现单词 respiratory，但会出现 apnea/hypopnea，也放行
            if not (RE_APNEA.search(text) or RE_HYPO.search(text)):
                continue

        # 直接命中 hypopnea
        if RE_HYPO.search(text):
            counts["hypopnea"] += 1
            total += 1
            continue

        # apnea 系
        if RE_APNEA.search(text):
            if include_central_mixed and (RE_MIXED.search(text) or RE_MA.search(text)):
                counts["mixed apnea"] += 1
                total += 1
                continue
            if include_central_mixed and (RE_CENT.search(text) or RE_CA.search(text)):
                counts["central apnea"] += 1
                total += 1
                continue
            if RE_OBSTR.search(text) or RE_OA.search(text):
                counts["obstructive apnea"] += 1
                total += 1
                continue
            # 只有“apnea”不带细分：保险起见，默认算 obstructive 也可以（很多数据默认指 OSA）
            # 你更谨慎的话就跳过；这里给出一个开关：
            # counts["obstructive apnea"] += 1; total += 1; continue

    counts = {k: v for k, v in counts.items() if v > 0}
    return total, counts

def severity_from_ahi(ahi: float) -> int:
    for i, (lo, hi) in enumerate(SEVERITY_BINS):
        if lo <= ahi < hi:
            return i
    return len(SEVERITY_BINS) - 1

def collect_all_texts(node: ET.Element) -> str:
    """收集事件节点及其所有后代节点的文本与有意义的属性，拼成一个大串用于关键词匹配。"""
    buf = []

    def add(txt):
        if txt:
            t = str(txt).strip()
            if t:
                buf.append(t)

    # 当前节点文本
    add(node.text)

    # 属性
    for k, v in (node.attrib or {}).items():
        add(k)
        add(v)

    # 广度遍历所有后代
    for ch in node.iter():
        # 标签名
        add(localname(ch.tag))
        # 文本
        add(ch.text)
        # 子节点中常见字段优先抓
        for key in ("EventConcept", "Concept", "EventSubType", "SubType", "Name", "EventName",
                    "EventType", "EventFamily", "Type", "Family", "Description", "Notes", "Note"):
            for cand in ch.findall(f".//{key}"):
                add(cand.text)
        # 参数类
        for param in ch.findall(".//Parameter"):
            add(param.text)
            for sub in param:
                add(sub.tag)
                add(sub.text)

        # 属性
        for k, v in (ch.attrib or {}).items():
            add(k)
            add(v)

        # tail 文本
        add(ch.tail)

    return " ".join(buf)


# ====================== RML 扫描 & 过滤 ======================
def scan_rml_candidates(root_dir: str) -> Dict[str, str]:
    """递归查找 *.rml，返回 {subject_id: path}；若文件名无数字，则用父目录名兜底。"""
    out: Dict[str, str] = {}
    root_dir = str(root_dir)
    for d, _, files in os.walk(root_dir):
        for fn in files:
            if not fn.lower().endswith(".rml"):
                continue
            full = os.path.join(d, fn)
            m = re.search(r"(\d{5,})", fn)  # 先从文件名找
            if m:
                sid = m.group(1)
            else:
                parent = os.path.basename(d)
                m2 = re.search(r"(\d{5,})", parent)
                sid = m2.group(1) if m2 else None
            if sid:
                out[sid] = full
    return out


def choose_rml_map(paths_root: str) -> Dict[str, str]:
    """优先 APNEA_RML_clean，再 APNEA_RML；都没有则在 paths_root 全局扫描"""
    clean_dir = os.path.join(paths_root, "APNEA_RML_clean")
    reg_dir = os.path.join(paths_root, "APNEA_RML")
    merged: Dict[str, str] = {}
    if os.path.isdir(reg_dir):
        merged.update(scan_rml_candidates(reg_dir))
    if os.path.isdir(clean_dir):
        merged.update(scan_rml_candidates(clean_dir))  # clean 覆盖
    if merged:
        return merged
    return scan_rml_candidates(paths_root)


def filter_map_by_ids_loose(rml_map: Dict[str, str], ids_filter: set) -> Dict[str, str]:
    """宽松过滤：允许 ids 出现在文件名或父目录名里"""
    out = {}
    for sid, path in rml_map.items():
        base = os.path.basename(path)
        parent = os.path.basename(os.path.dirname(path))
        for want in ids_filter:
            if want == sid or want in base or want in parent:
                out[sid] = path
                break
    return out


# ====================== 主流程 ======================
def main():
    ap = argparse.ArgumentParser(description="从 YAML 配置生成 PSG-Audio 的夜级真值 AHI & 严重度 CSV")
    ap.add_argument("--config", default=None, help="配置文件路径（YAML），默认自动寻找 config.yaml")
    ap.add_argument("--rml-dir", default=None, help="RML 根目录（优先级最高，例如：/.../data/preprocess/rml）")
    ap.add_argument("--out-csv", default=None, help="输出 CSV（默认：<paths.root>/processed/labels/night_level_ahi.csv）")
    ap.add_argument("--fallback-record-hours", type=float, default=8.0, help="无法估计 TST 时的兜底小时数（默认 8h）")
    ap.add_argument("--exclude-central-mixed", action="store_true", help="计算 AHI 时不计入 central/mixed apnea")
    args = ap.parse_args()

    cfg_path = resolve_config_path(args.config)
    cfg = load_config(cfg_path)

    # 解析 paths.root：相对路径相对于 config.yaml 所在目录
    cfg_dir = Path(cfg_path).resolve().parent
    paths_root_cfg = cfg.get("paths", {}).get("root", "data")
    paths_root = str((cfg_dir / paths_root_cfg).resolve()) if not os.path.isabs(paths_root_cfg) else paths_root_cfg
    print(f"[Info] 解析后的 paths.root = {paths_root}")

    # 选择 RML 来源
    if args.rml_dir and os.path.isdir(args.rml_dir):
        print(f"[Info] 使用 --rml-dir：{args.rml_dir}")
        rml_map = scan_rml_candidates(args.rml_dir)
    else:
        rml_map = choose_rml_map(paths_root)
        print(f"[Info] 基于 paths.root 搜索 RML：{paths_root}")

    if not rml_map:
        raise SystemExit(f"未在 {paths_root} 或指定目录下找到任何 RML 文件。")

    # ids 过滤（从 config.data.ids 读取）
    ids_filter = set(cfg.get("data", {}).get("ids") or [])
    if ids_filter:
        filtered = filter_map_by_ids_loose(rml_map, ids_filter)
        if not filtered:
            avail = sorted(rml_map.keys(), key=lambda x: int(x))
            raise SystemExit(
                f"按 data.ids 过滤后为空：{sorted(ids_filter)}\n"
                f"可用受试者ID样例：{avail[:20]}{' ...' if len(avail)>20 else ''}"
            )
        rml_map = filtered
        print(f"[Info] 由 data.ids 过滤：{sorted(rml_map.keys(), key=lambda x: int(x))}")

    # 输出路径
    out_csv = args.out_csv or os.path.join(paths_root, "processed", "labels", "night_level_ahi.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    include_central_mixed = not args.exclude_central_mixed
    results: Dict[str, Dict] = {}

    # 逐 subject 解析与统计
    for sid in sorted(rml_map.keys(), key=lambda x: int(x)):
        rml_path = rml_map[sid]
        events = parse_rml_file(rml_path)
        if not events:
            print(f"[WARN] 无事件：{rml_path}")
            continue

        # 先看标签分布（调试用，看完可以注释掉）
        print(f"\n[DEBUG] Subject {sid} label histogram from: {rml_path}")
        debug_label_hist(events, topk=60)
        tst = compute_tst_seconds_from_events(events)
        if tst is None or tst <= 0:
            tst = max(1e-6, args.fallback_record_hours * 3600.0)

        n_resp, counts = count_respiratory_events(events, include_central_mixed=include_central_mixed)
        hours = max(1e-6, tst / 3600.0)
        ahi = n_resp / hours
        sev = severity_from_ahi(ahi)

        results[sid] = {
            "ahi": float(ahi),
            "severity": int(sev),
            "severity_name": SEVERITY_NAMES[sev],
            "n_resp_events": int(n_resp),
            "total_seconds": float(tst),
            "total_hours": float(hours),
            "resp_counts": counts,
            "rml_path": rml_path,
        }

    if not results:
        raise SystemExit("没有任何受试者成功生成 AHI/Severity，请检查 RML 内容与解析逻辑。")

    # 写 CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "subject_id", "true_ahi", "true_severity", "severity_name",
            "n_resp_events", "total_seconds", "total_hours", "resp_counts_json", "rml_path"
        ])
        for sid in sorted(results.keys(), key=lambda x: int(x)):
            row = results[sid]
            w.writerow([
                sid,
                f"{row['ahi']:.6f}",
                row["severity"],
                row["severity_name"],
                row["n_resp_events"],
                f"{row['total_seconds']:.3f}",
                f"{row['total_hours']:.6f}",
                json.dumps(row["resp_counts"], ensure_ascii=False, sort_keys=True),
                row["rml_path"],
            ])

    print(f"[OK] 写出到: {out_csv} （{len(results)} subjects）")
    print(f"[Info] include_central_mixed = {include_central_mixed}")


if __name__ == "__main__":
    main()
