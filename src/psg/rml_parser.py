# rml_parser.py
import os
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional


class RMLParser:
    """
    鲁棒解析 PSG-Audio RML（XML/纯文本）：
    - 命名空间无关；整树扫描
    - 事件字段可来自：子节点文本 或 XML 属性
    - 时间支持：秒/毫秒/hh:mm:ss(.ms)/ISO8601；若疑似样本点且容器里有采样率，则自动转秒
    - 事件名做别名归一到 event_mapping 的键
    返回: [{"type":<规范名>,"start":sec,"end":sec,"label":int}, ...]
    """

    _TYPE_FIELDS   = {"eventtype", "eventconcept", "name", "annotation", "type", "category"}
    _START_FIELDS  = {"start", "starttime", "begin", "position"}
    _DUR_FIELDS    = {"duration", "durationms", "durationsec"}
    _END_FIELDS    = {"end", "stop", "finishtime"}
    _FS_FIELDS     = {"samplerate", "samplingrate", "fs", "hz"}  # 用于样本点→秒的启发式转换

    def __init__(self, event_mapping: Dict[str, int], debug: bool = False):
        self.event_mapping = event_mapping
        self.default_label = event_mapping.get("Normal", 0)
        self.debug = debug

        # PSG-Audio 常见事件名别名（key: 归一小写去空格/下划线/短横线；value: mapping 中的规范名）
        base_aliases = {
            # Respiratory
            "obstructiveapnea": "ObstructiveApnea",
            "obstructive apnea": "ObstructiveApnea",
            "oa": "ObstructiveApnea",
            "centralapnea": "CentralApnea",
            "central apnea": "CentralApnea",
            "mixedapnea": "MixedApnea",
            "mixed apnea": "MixedApnea",
            "hypopnea": "Hypopnea",
            "periodicrespiration": "PeriodicRespiration",
            "cheynestokesrespiration": "PeriodicRespiration",

            # RERA（如需单列，把下面两行改为 "RERA"）
            "respiratoryeffortrelatedarousal": "Arousal",
            "rera": "Arousal",

            # Neurological
            "arousal": "Arousal",

            # Limb
            "legmovement": "LegMovement",

            # Nasal
            "snore": "Snore",

            # Cardiac
            "bradycardia": "Bradycardia",
            "tachycardia": "Tachycardia",
            "longrr": "LongRR",
            "pttdrop": "PttDrop",
            "heartratedrop": "Bradycardia",
            "heartraterise": "Tachycardia",
            "asystole": "Bradycardia",
            "atrialfibrilation": "Tachycardia",
            "sinustachycardia": "Tachycardia",
            "narrowcomplextachycardia": "Tachycardia",
            "widecomplextachycardia": "Tachycardia",

            # SpO2
            "relativedesaturation": "RelativeDesaturation",
            "absolutedesaturation": "RelativeDesaturation",

            # Other
            "normal": "Normal",
        }
        self.aliases = {k: v for k, v in base_aliases.items() if v in self.event_mapping}

    # ----------------- 公有入口 -----------------
    def parse_rml(self, rml_path: str) -> List[Dict]:
        if not os.path.exists(rml_path):
            print(f"[RML] file not found: {rml_path}")
            return []

        # 先试 XML
        try:
            events = self._parse_xml_tree_scan(rml_path)
            if len(events) > 0:
                if self.debug:
                    print(f"[RML] parsed {len(events)} events (XML-tree-scan)")
                return events
            else:
                print("[RML] XML parsed 0 events, fallback to plaintext...")
        except Exception as e:
            print(f"[RML] XML parse error: {e}; fallback to plaintext...")

        # 兜底：纯文本
        return self._parse_plaintext(rml_path)

    # ----------------- XML 解析（整树扫描 + 属性/文本统一收集） -----------------
    def _parse_xml_tree_scan(self, rml_path: str) -> List[Dict]:
        root = ET.parse(rml_path).getroot()
        events, seen = [], set()

        for node in root.iter():
            bucket = self._collect_fields_text_and_attrs(node)   # 文本+属性
            if self.debug and not bucket:
                continue

            # 类型
            type_txt = self._pick_first(bucket, self._TYPE_FIELDS)
            if not type_txt:
                continue

            # 起始
            start = self._pick_time(bucket, self._START_FIELDS)
            if start is None:
                continue

            # 时长/结束
            dur = self._pick_time(bucket, self._DUR_FIELDS)
            end = self._pick_time(bucket, self._END_FIELDS)

            # 若看起来是“样本点”，尝试用容器里的采样率换算
            # 简单启发：start 很大(>1e5)、dur/end 也大、且 bucket 里有采样率字段
            if (end is None and dur is not None) or (end is not None):
                if (start is not None) and (start > 1e5 or (dur and dur > 1e5) or (end and end > 1e5)):
                    fs = self._pick_time(bucket, self._FS_FIELDS)
                    if fs and fs > 0:
                        start = start / fs if start else start
                        if dur is not None:
                            dur = dur / fs
                        if end is not None:
                            end = end / fs

            if end is None and dur is not None:
                end = start + dur
            if end is None or end <= start:
                continue

            canon = self._canon(type_txt)
            if not canon or canon not in self.event_mapping:
                continue

            key = (canon, round(start, 3), round(end, 3))
            if key in seen:
                continue
            seen.add(key)

            events.append({
                "type": canon,
                "start": float(start),
                "end": float(end),
                "label": self.event_mapping[canon],
            })

        events.sort(key=lambda x: x["start"])
        return events

    # ----------------- 纯文本兜底解析 -----------------
    def _parse_plaintext(self, rml_path: str) -> List[Dict]:
        events = []
        with open(rml_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                m1 = re.match(r"(.+?)\s+(\d+\.?\d*)\s+(\d+\.?\d*)", line)
                m2 = re.match(r"(\d+\.?\d*)\s+(\d+\.?\d*)\s+(.+)", line)
                if m1:
                    etype, s, e = m1.group(1).strip(), float(m1.group(2)), float(m1.group(3))
                elif m2:
                    s, e, etype = float(m2.group(1)), float(m2.group(2)), m2.group(3).strip()
                else:
                    continue

                canon = self._canon(etype)
                if canon and e > s:
                    events.append({"type": canon, "start": s, "end": e, "label": self.event_mapping[canon]})

        events.sort(key=lambda x: x["start"])
        if self.debug:
            print(f"[RML] parsed {len(events)} events (PLAIN)")
        return events

    # ----------------- 工具方法 -----------------
    @staticmethod
    def _local(tag: str) -> str:
        return tag.split('}', 1)[-1] if '}' in tag else tag

    @staticmethod
    def _norm(name: Optional[str]) -> str:
        return re.sub(r"[\s_\-()]+", "", (name or "").lower())

    def _canon(self, raw_type: str) -> str:
        key = self._norm(raw_type)
        if key in self.aliases:
            return self.aliases[key]
        for k in self.event_mapping.keys():
            if self._norm(k) == key:
                return k
        return ""

    def _collect_fields_text_and_attrs(self, node) -> Dict[str, str]:
        """
        收集 node 子树中的“本地标签名 -> 文本”，以及“本地标签名.属性名 -> 属性值”。
        e.g. <ScoredEvent Start="12.5" Duration="10" EventType="Obstructive Apnea">
               <Start>12.5</Start><Duration>10</Duration>...
             -> {"start":"12.5","duration":"10","eventtype":"Obstructive Apnea"}
        """
        bucket: Dict[str, str] = {}
        for e in node.iter():
            nm = self._local(e.tag).lower()
            # 文本
            tx = (e.text or "").strip()
            if tx and nm not in bucket:
                bucket[nm] = tx
            # 属性
            for attr_name, attr_val in (e.attrib or {}).items():
                an = f"{nm}.{attr_name}".lower()
                if attr_val and an not in bucket:
                    bucket[an] = str(attr_val).strip()

        # 把 “标签.属性” 也映射成裸名（属性名）以便被字段名集合匹配
        # 如 ScoredEvent Start=".." -> 同时提供 "scoredevent.start" 和 "start"
        extra = {}
        for k, v in bucket.items():
            if "." in k:
                bare = k.split(".", 1)[1]
                if bare not in bucket:
                    extra[bare] = v
        bucket.update(extra)
        return bucket

    def _pick_first(self, bucket: Dict[str, str], names: set) -> Optional[str]:
        # 直接匹配键
        for k in names:
            if k in bucket and bucket[k]:
                return bucket[k]
        # 含义匹配：键里包含候选名（防止出现前后缀）
        for key in bucket.keys():
            if any(k in key for k in names):
                if bucket[key]:
                    return bucket[key]
        return None

    def _pick_time(self, bucket: Dict[str, str], names: set) -> Optional[float]:
        # 直接匹配
        for k in names:
            if k in bucket and bucket[k]:
                sec = self._parse_time_to_seconds(bucket[k])
                if sec is not None:
                    return sec
        # 含义匹配
        for key in bucket.keys():
            if any(k in key for k in names):
                sec = self._parse_time_to_seconds(bucket[key])
                if sec is not None:
                    return sec
        return None

    @staticmethod
    def _parse_time_to_seconds(txt: Optional[str]) -> Optional[float]:
        if not txt:
            return None
        t = txt.strip()

        # 1) 纯数字（优先按秒）
        try:
            return float(t)
        except Exception:
            pass

        # 2) hh:mm:ss(.ms) 或 mm:ss
        if ":" in t:
            parts = t.split(":")
            try:
                parts = [float(x) for x in parts]
                if len(parts) == 2:
                    mm, ss = parts
                    return mm * 60.0 + ss
                elif len(parts) == 3:
                    hh, mm, ss = parts
                    return hh * 3600.0 + mm * 60.0 + ss
            except Exception:
                pass

        # 3) ISO8601 PTxxHxxMxxS
        if t.upper().startswith("PT"):
            h = re.search(r'(\d+(?:\.\d+)?)H', t, re.I)
            m = re.search(r'(\d+(?:\.\d+)?)M', t, re.I)
            s = re.search(r'(\d+(?:\.\d+)?)S', t, re.I)
            sec = 0.0
            if h: sec += float(h.group(1)) * 3600.0
            if m: sec += float(m.group(1)) * 60.0
            if s: sec += float(s.group(1))
            return sec if sec > 0 else None

        # 4) 纯整数且很大：可能是毫秒（先按 ms）
        if t.isdigit():
            try:
                v = int(t)
                if v > 10000:
                    return v / 1000.0
            except Exception:
                pass

        return None
