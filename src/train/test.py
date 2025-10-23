import pickle
import os

data_dir = "/Users/liyuxiang/Downloads/sleep-apnea-main/data/processed/signals"
labels = set()
for filename in os.listdir(data_dir):
    if filename.endswith(".pickle"):
        with open(os.path.join(data_dir, filename), 'rb') as f:
            events = pickle.load(f)
            for ev in events:
                labels.add(str(ev.label))
print("实际标签类别：", sorted(labels))
print("标签数量：", len(labels))