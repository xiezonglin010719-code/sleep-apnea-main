# # # -*- coding: utf-8 -*-
# # import os
# # import sys
# # import xml.etree.ElementTree as ET
# # from collections import Counter, defaultdict
# #
# # RML_ROOT = "/Users/liyuxiang/Downloads/sleep-apnea-main/data/preprocess/rml"  # 放到 rml 根目录（不是某个病人子目录）
# #
# # def strip_ns(tag: str) -> str:
# #     """去掉XML命名空间前缀，返回裸标签名"""
# #     if tag is None:
# #         return ""
# #     if "}" in tag:
# #         return tag.split("}", 1)[1]
# #     if ":" in tag:
# #         return tag.split(":", 1)[1]
# #     return tag
# #
# # def get_text(elem, *candidates):
# #     """从 elem 下尝试取若干子元素文本（大小写/命名空间不敏感）"""
# #     for cand in candidates:
# #         # cand 可能是 'EventType' / 'eventtype' / 'Type' 等
# #         for child in elem.iter():
# #             if strip_ns(child.tag).lower() == cand.lower():
# #                 if child.text and child.text.strip():
# #                     return child.text.strip()
# #     return None
# #
# # def get_attr(elem, *candidates):
# #     """尝试从属性里取 Name/Label 等"""
# #     for key in candidates:
# #         for k, v in elem.attrib.items():
# #             if k.lower() == key.lower() and v and v.strip():
# #                 return v.strip()
# #     return None
# #
# # def collect_labels_from_file(path: str) -> list:
# #     """
# #     尽可能兼容多种RML结构：
# #       - <ScoredEvent><EventType> / <EventConcept>
# #       - <Event Name="..."> 或 <Event><Type> / <Label>
# #       - 命名空间前缀
# #     返回：该文件提取到的标签字符串列表
# #     """
# #     labels = []
# #     try:
# #         tree = ET.parse(path)
# #         root = tree.getroot()
# #
# #         # 1) 优先扫常见 AASM 结构：ScoredEvent
# #         scored_events = []
# #         for node in root.iter():
# #             if strip_ns(node.tag).lower() == "scoredevent":
# #                 scored_events.append(node)
# #
# #         if scored_events:
# #             for ev in scored_events:
# #                 # 典型：EventConcept 是具体类型，EventType 是大类（Respiratory/Arousal等）
# #                 concept = get_text(ev, "EventConcept", "Concept", "Name", "Label")
# #                 if concept:
# #                     labels.append(concept)
# #                     continue
# #                 # 退化到 EventType
# #                 etype = get_text(ev, "EventType", "Type")
# #                 if etype:
# #                     labels.append(etype)
# #                     continue
# #                 # 属性里找一找
# #                 name_attr = get_attr(ev, "Name", "Label", "Type")
# #                 if name_attr:
# #                     labels.append(name_attr)
# #             return labels
# #
# #         # 2) 其他结构：Event / EventList / Annotation
# #         #   尽量宽松抓：任何节点名含 'event' 的都试
# #         event_like_nodes = []
# #         for node in root.iter():
# #             t = strip_ns(node.tag).lower()
# #             if "event" in t or t in ("annotation", "scoredannotation"):
# #                 event_like_nodes.append(node)
# #
# #         if event_like_nodes:
# #             for ev in event_like_nodes:
# #                 # 先从属性取
# #                 name_attr = get_attr(ev, "Name", "Label", "Type", "EventType", "EventConcept")
# #                 if name_attr:
# #                     labels.append(name_attr)
# #                     continue
# #                 # 再从子节点取
# #                 txt = get_text(ev, "EventConcept", "Concept", "EventType", "Type", "Label", "Name")
# #                 if txt:
# #                     labels.append(txt)
# #             return labels
# #
# #         # 3) 兜底：打印该文件的根结构（帮助你调格式）
# #         print(f"[HINT] {os.path.basename(path)} 未匹配到已知结构。根标签: {strip_ns(root.tag)}，"
# #               f"子标签示例: {[strip_ns(c.tag) for c in list(root)[:5]]}")
# #         return labels
# #
# #     except Exception as e:
# #         print(f"[WARN] 解析失败 {os.path.basename(path)}: {e}")
# #         return labels
# #
# # def main():
# #     if not os.path.isdir(RML_ROOT):
# #         print(f"[ERROR] 目录不存在: {RML_ROOT}")
# #         sys.exit(1)
# #
# #     all_labels = Counter()
# #     per_file = defaultdict(Counter)
# #     files = []
# #     for base, _, names in os.walk(RML_ROOT):
# #         for fn in names:
# #             if fn.lower().endswith(".rml") or fn.lower().endswith(".xml"):
# #                 files.append(os.path.join(base, fn))
# #
# #     if not files:
# #         print(f"[ERROR] 在 {RML_ROOT} 下未找到任何 .rml/.xml")
# #         sys.exit(1)
# #
# #     for p in files:
# #         lbls = collect_labels_from_file(p)
# #         if not lbls:
# #             # 给点线索，帮助我们再定制
# #             try:
# #                 root = ET.parse(p).getroot()
# #                 head = [strip_ns(c.tag) for c in list(root)[:10]]
# #                 print(f"[INFO] {os.path.basename(p)} 没抓到标签。根: {strip_ns(root.tag)}，前几级子节点: {head}")
# #             except Exception:
# #                 pass
# #         else:
# #             per_file[os.path.basename(p)].update(lbls)
# #             all_labels.update(lbls)
# #
# #     print("\nRML 中发现的所有事件标签（去重）:")
# #     if not all_labels:
# #         print("  （空）→ 大概率是命名空间或结构未覆盖；上面已打印了文件的根和子节点结构，发我看下即可扩展解析。")
# #     else:
# #         for name, cnt in all_labels.most_common():
# #             print(f" - {name}  ({cnt} 次)")
# #
# #     print("\n样例文件内的标签统计（前 5 个文件）:")
# #     for i, (fn, cnts) in enumerate(per_file.items()):
# #         if i >= 5: break
# #         print(f"  {fn}: " + ", ".join([f"{k}:{v}" for k, v in cnts.most_common()]))
# #
# # if __name__ == "__main__":
# #     main()
#
#
# # 一次性检查所有客户端 train/val 标签分布
# # import os, numpy as np, glob
# # root = "/Users/liyuxiang/Downloads/sleep-apnea-main/src/federated/data/clients"
# # for d in sorted(glob.glob(os.path.join(root, "*"))):
# #     tr = os.path.join(d, "train.npz")
# #     va = os.path.join(d, "val.npz")
# #     if os.path.exists(tr):
# #         y = np.load(tr)["labels"]
# #         cls, cnt = np.unique(y, return_counts=True)
# #         print(f"[{os.path.basename(d)}] train:", dict(zip(cls.tolist(), cnt.tolist())))
# #     if os.path.exists(va):
# #         y = np.load(va)["labels"]
# #         cls, cnt = np.unique(y, return_counts=True)
# #         print(f"[{os.path.basename(d)}]   val:", dict(zip(cls.tolist(), cnt.tolist())))
# #
# # import numpy as np
# # from sklearn.decomposition import PCA
# # import matplotlib.pyplot as plt
# #
# # data = np.load("/Users/liyuxiang/Downloads/sleep-apnea-main/src/federated/data/preprocess/psg_features/00000995_features.npz")
# # X, y = data["features"].reshape(len(data["features"]), -1), data["labels"]
# #
# # pca = PCA(n_components=2)
# # X_2d = pca.fit_transform(X)
# #
# # plt.figure(figsize=(6,5))
# # for label, color in zip([0,1,2], ["gray","red","blue"]):
# #     plt.scatter(X_2d[y==label,0], X_2d[y==label,1], c=color, s=5, label=str(label))
# # plt.legend(); plt.show()
#
# # -*- coding: utf-8 -*-
#
# # inspect_pickles.py
# -*- coding: utf-8 -*-
# import os
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from src.apnea_events.apnea_event import ApneaEvent  # 导入你的事件类
#
#
# def load_all_events(pickle_dir):
#     """加载指定目录下所有.pickle文件中的事件数据，按标签分组"""
#     label_events = defaultdict(list)  # 格式：{标签: [事件1, 事件2, ...]}
#
#     # 遍历目录下所有.pickle文件
#     for filename in os.listdir(pickle_dir):
#         if filename.endswith(".pickle"):
#             file_path = os.path.join(pickle_dir, filename)
#             with open(file_path, 'rb') as f:
#                 events = pickle.load(f)  # 加载事件列表（ApneaEvent对象）
#
#             # 按标签分组
#             for event in events:
#                 if isinstance(event, ApneaEvent):
#                     label = event.label  # 获取事件标签（如"normal"、"ObstructiveApnea"）
#                     label_events[label].append(event)
#
#     print(f"加载完成！标签种类：{list(label_events.keys())}")
#     for label, events in label_events.items():
#         print(f"  标签 {label}：{len(events)} 个事件")
#     return label_events
#
#
# def visualize_label_features(label_events, n_samples=5, figsize=(15, 10)):
#     """可视化不同标签的梅尔频谱图（特征）差异"""
#     labels = list(label_events.keys())
#     if len(labels) < 2:
#         print("至少需要2种标签才能对比")
#         return
#
#     # 每种标签随机选n_samples个样本可视化
#     for i in range(n_samples):
#         plt.figure(figsize=figsize)
#         for idx, label in enumerate(labels):
#             # 随机选一个事件
#             event = np.random.choice(label_events[label])
#             signal = event.signal  # 这是预处理后的梅尔频谱图（图像特征）
#
#             plt.subplot(1, len(labels), idx + 1)
#             plt.imshow(signal, cmap='viridis')  # 梅尔频谱图常用viridis色图
#             plt.title(f"标签：{label}\n事件索引：{event.index}")
#             plt.axis('off')
#         plt.tight_layout()
#         plt.show()
#
#
# def analyze_feature_stats(label_events):
#     """分析不同标签的特征统计差异（均值、标准差等）"""
#     stats = {}
#     for label, events in label_events.items():
#         # 提取所有事件的特征（假设signal是二维数组：[H, W]）
#         features = np.array([event.signal for event in events])  # 形状：[样本数, H, W]
#         # 计算统计量
#         stats[label] = {
#             "mean": np.mean(features),
#             "std": np.std(features),
#             "max": np.max(features),
#             "min": np.min(features),
#             "shape": features.shape  # 特征尺寸和样本数
#         }
#
#     # 打印统计结果
#     print("\n特征统计对比：")
#     for label, stat in stats.items():
#         print(f"\n标签 {label}：")
#         print(f"  样本数：{stat['shape'][0]}")
#         print(f"  特征尺寸：{stat['shape'][1:]}")
#         print(f"  均值：{stat['mean']:.2f}")
#         print(f"  标准差：{stat['std']:.2f}")
#         print(f"  最大值：{stat['max']:.2f}")
#         print(f"  最小值：{stat['min']:.2f}")
#
#
# if __name__ == "__main__":
#     # 配置：替换为你的pickle文件保存目录（即Processor类中signals_path配置的路径）
#     PICKLE_DIR = "/Users/liyuxiang/Downloads/sleep-apnea-main/data/processed/signals"
#
#     # 步骤1：加载所有事件并按标签分组
#     label_events = load_all_events(PICKLE_DIR)
#
#     # 步骤2：可视化不同标签的特征（梅尔频谱图）
#     visualize_label_features(label_events, n_samples=3)  # 对比3组样本
#
#     # 步骤3：分析特征的统计差异
#     analyze_feature_stats(label_events)
import os
import xml.dom.minidom
from collections import Counter
from pathlib import Path


def print_rml_event_types(rml_folder: str):
    """
    遍历指定目录下所有 RML 文件，打印所有 <Event> 标签的 Type 属性
    :param rml_folder: RML 文件所在的根目录（比如你的 rml_preprocess_path）
    """
    all_event_types = []  # 存储所有事件类型
    rml_files = []

    # 1. 查找目录下所有 RML 文件（包括子目录）
    for root, _, files in os.walk(rml_folder):
        for file in files:
            if file.endswith(".rml"):
                rml_files.append(os.path.join(root, file))

    if not rml_files:
        print(f"⚠️  在目录 {rml_folder} 中未找到任何 .rml 文件")
        return

    print(f"📊 找到 {len(rml_files)} 个 RML 文件，开始提取 Event Type...\n")

    # 2. 逐个解析 RML 文件，提取 Type 属性
    for rml_file in rml_files:
        file_name = Path(rml_file).name
        print(f"=== 处理文件：{file_name} ===")

        try:
            # 解析 RML 文件
            domtree = xml.dom.minidom.parse(rml_file)
            group = domtree.documentElement
            events = group.getElementsByTagName('Event')  # 提取所有 <Event> 标签

            if not events:
                print(f"  该文件中没有 <Event> 标签")
                continue

            # 提取当前文件的所有 Event Type
            file_event_types = []
            for event in events:
                event_type = event.getAttribute('Type')  # 核心：获取 Type 属性
                file_event_types.append(event_type)
                all_event_types.append(event_type)
                # 可选：同时打印该事件的 Start 时间戳
                event_start = event.getAttribute('Start')
                print(f"  Type: {event_type:<20} | Start 时间: {event_start}s")

            print(f"  该文件事件类型统计：{dict(Counter(file_event_types))}\n")

        except Exception as e:
            print(f"  ❌ 解析文件 {file_name} 出错：{str(e)}\n")
            continue

    # 3. 统计所有文件的事件类型总分布
    print("=" * 50)
    print("📈 所有 RML 文件的 Event Type 总统计：")
    type_count = Counter(all_event_types)
    for event_type, count in type_count.most_common():
        print(f"  {event_type:<20} 出现次数：{count}")


if __name__ == "__main__":
    # --------------------------
    # 配置：替换为你的 RML 文件目录
    # --------------------------
    # 比如你的 rml_preprocess_path 或 rml_download_path
    RML_FOLDER = "/Users/liyuxiang/Downloads/sleep-apnea-main/data/preprocess/rml/00001245"

    # 运行脚本
    print_rml_event_types(RML_FOLDER)