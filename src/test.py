# # -*- coding: utf-8 -*-
# import os
# import sys
# import xml.etree.ElementTree as ET
# from collections import Counter, defaultdict
#
# RML_ROOT = "/Users/liyuxiang/Downloads/sleep-apnea-main/data/preprocess/rml"  # 放到 rml 根目录（不是某个病人子目录）
#
# def strip_ns(tag: str) -> str:
#     """去掉XML命名空间前缀，返回裸标签名"""
#     if tag is None:
#         return ""
#     if "}" in tag:
#         return tag.split("}", 1)[1]
#     if ":" in tag:
#         return tag.split(":", 1)[1]
#     return tag
#
# def get_text(elem, *candidates):
#     """从 elem 下尝试取若干子元素文本（大小写/命名空间不敏感）"""
#     for cand in candidates:
#         # cand 可能是 'EventType' / 'eventtype' / 'Type' 等
#         for child in elem.iter():
#             if strip_ns(child.tag).lower() == cand.lower():
#                 if child.text and child.text.strip():
#                     return child.text.strip()
#     return None
#
# def get_attr(elem, *candidates):
#     """尝试从属性里取 Name/Label 等"""
#     for key in candidates:
#         for k, v in elem.attrib.items():
#             if k.lower() == key.lower() and v and v.strip():
#                 return v.strip()
#     return None
#
# def collect_labels_from_file(path: str) -> list:
#     """
#     尽可能兼容多种RML结构：
#       - <ScoredEvent><EventType> / <EventConcept>
#       - <Event Name="..."> 或 <Event><Type> / <Label>
#       - 命名空间前缀
#     返回：该文件提取到的标签字符串列表
#     """
#     labels = []
#     try:
#         tree = ET.parse(path)
#         root = tree.getroot()
#
#         # 1) 优先扫常见 AASM 结构：ScoredEvent
#         scored_events = []
#         for node in root.iter():
#             if strip_ns(node.tag).lower() == "scoredevent":
#                 scored_events.append(node)
#
#         if scored_events:
#             for ev in scored_events:
#                 # 典型：EventConcept 是具体类型，EventType 是大类（Respiratory/Arousal等）
#                 concept = get_text(ev, "EventConcept", "Concept", "Name", "Label")
#                 if concept:
#                     labels.append(concept)
#                     continue
#                 # 退化到 EventType
#                 etype = get_text(ev, "EventType", "Type")
#                 if etype:
#                     labels.append(etype)
#                     continue
#                 # 属性里找一找
#                 name_attr = get_attr(ev, "Name", "Label", "Type")
#                 if name_attr:
#                     labels.append(name_attr)
#             return labels
#
#         # 2) 其他结构：Event / EventList / Annotation
#         #   尽量宽松抓：任何节点名含 'event' 的都试
#         event_like_nodes = []
#         for node in root.iter():
#             t = strip_ns(node.tag).lower()
#             if "event" in t or t in ("annotation", "scoredannotation"):
#                 event_like_nodes.append(node)
#
#         if event_like_nodes:
#             for ev in event_like_nodes:
#                 # 先从属性取
#                 name_attr = get_attr(ev, "Name", "Label", "Type", "EventType", "EventConcept")
#                 if name_attr:
#                     labels.append(name_attr)
#                     continue
#                 # 再从子节点取
#                 txt = get_text(ev, "EventConcept", "Concept", "EventType", "Type", "Label", "Name")
#                 if txt:
#                     labels.append(txt)
#             return labels
#
#         # 3) 兜底：打印该文件的根结构（帮助你调格式）
#         print(f"[HINT] {os.path.basename(path)} 未匹配到已知结构。根标签: {strip_ns(root.tag)}，"
#               f"子标签示例: {[strip_ns(c.tag) for c in list(root)[:5]]}")
#         return labels
#
#     except Exception as e:
#         print(f"[WARN] 解析失败 {os.path.basename(path)}: {e}")
#         return labels
#
# def main():
#     if not os.path.isdir(RML_ROOT):
#         print(f"[ERROR] 目录不存在: {RML_ROOT}")
#         sys.exit(1)
#
#     all_labels = Counter()
#     per_file = defaultdict(Counter)
#     files = []
#     for base, _, names in os.walk(RML_ROOT):
#         for fn in names:
#             if fn.lower().endswith(".rml") or fn.lower().endswith(".xml"):
#                 files.append(os.path.join(base, fn))
#
#     if not files:
#         print(f"[ERROR] 在 {RML_ROOT} 下未找到任何 .rml/.xml")
#         sys.exit(1)
#
#     for p in files:
#         lbls = collect_labels_from_file(p)
#         if not lbls:
#             # 给点线索，帮助我们再定制
#             try:
#                 root = ET.parse(p).getroot()
#                 head = [strip_ns(c.tag) for c in list(root)[:10]]
#                 print(f"[INFO] {os.path.basename(p)} 没抓到标签。根: {strip_ns(root.tag)}，前几级子节点: {head}")
#             except Exception:
#                 pass
#         else:
#             per_file[os.path.basename(p)].update(lbls)
#             all_labels.update(lbls)
#
#     print("\nRML 中发现的所有事件标签（去重）:")
#     if not all_labels:
#         print("  （空）→ 大概率是命名空间或结构未覆盖；上面已打印了文件的根和子节点结构，发我看下即可扩展解析。")
#     else:
#         for name, cnt in all_labels.most_common():
#             print(f" - {name}  ({cnt} 次)")
#
#     print("\n样例文件内的标签统计（前 5 个文件）:")
#     for i, (fn, cnts) in enumerate(per_file.items()):
#         if i >= 5: break
#         print(f"  {fn}: " + ", ".join([f"{k}:{v}" for k, v in cnts.most_common()]))
#
# if __name__ == "__main__":
#     main()


# 一次性检查所有客户端 train/val 标签分布
import os, numpy as np, glob
root = "/Users/liyuxiang/Downloads/sleep-apnea-main/src/federated/data/clients"
for d in sorted(glob.glob(os.path.join(root, "*"))):
    tr = os.path.join(d, "train.npz")
    va = os.path.join(d, "val.npz")
    if os.path.exists(tr):
        y = np.load(tr)["labels"]
        cls, cnt = np.unique(y, return_counts=True)
        print(f"[{os.path.basename(d)}] train:", dict(zip(cls.tolist(), cnt.tolist())))
    if os.path.exists(va):
        y = np.load(va)["labels"]
        cls, cnt = np.unique(y, return_counts=True)
        print(f"[{os.path.basename(d)}]   val:", dict(zip(cls.tolist(), cnt.tolist())))

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = np.load("/Users/liyuxiang/Downloads/sleep-apnea-main/src/federated/data/preprocess/psg_features/00000995_features.npz")
X, y = data["features"].reshape(len(data["features"]), -1), data["labels"]

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(6,5))
for label, color in zip([0,1,2], ["gray","red","blue"]):
    plt.scatter(X_2d[y==label,0], X_2d[y==label,1], c=color, s=5, label=str(label))
plt.legend(); plt.show()

# -*- coding: utf-8 -*-
