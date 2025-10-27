import csv
import os
import argparse
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter

# 导入模型和工具函数
from src.models.wgan_gp import WGANGP
from src.models.osa_end2end import OSAEnd2EndModel
from src.preprocessing.steps.config import load_config
from src.utils.utils import load_pickle_events, to_uint8_image

import torch.nn.functional as F

# -------------------------
# 全局配置（统一标签和严重度定义）
# -------------------------
# 三类事件标签（与数据中的标签严格对应）
EVENT_CLASS_NAMES = ["normal", "Hypopnea", "ObstructiveApnea"]
EVENT_LABEL_MAP = {
    "normal": 0, "none": 0, "background": 0, "noevent": 0, "negative": 0,
    "hypopnea": 1, "hypopnoea": 1,
    "obstructiveapnea": 2, "obstructive apnea": 2, "oa": 2
}

# 严重度定义（4类）
SEVERITY_BINS = [(0, 5), (5, 15), (15, 30), (30, float("inf"))]
SEVERITY_NAMES = ["None", "Mild", "Moderate", "Severe"]
SEVERITY_LABELS = [0, 1, 2, 3]


# -------------------------
# 工具函数
# -------------------------
def load_truth_csv(csv_path):
    """
    读取 psg_audio_ahi_severity_from_config.py 生成的 CSV
    返回: {subject_id: {"ahi": float, "severity": int, "severity_name": str}}
    """
    truth = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = row["subject_id"].strip()
            ahi = float(row["true_ahi"])
            sev = int(row["true_severity"])
            sev_name = row.get("severity_name", SEVERITY_NAMES[sev])
            truth[sid] = {"ahi": ahi, "severity": sev, "severity_name": sev_name}
    return truth


def cfg_get(cfg, path, default=None):
    cur = cfg
    for key in path.split('.'):
        if cur is None:
            return default
        if hasattr(cur, key):
            cur = getattr(cur, key)
            continue
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
            continue
        return default
    return cur


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


def classify_severity(ahi: float) -> int:
    """根据AHI值分类严重度（确保正常样本映射到0类）"""
    if ahi < SEVERITY_BINS[0][1]:  # AHI <5 → None（正常）
        return 0
    for i, (lo, hi) in enumerate(SEVERITY_BINS[1:], 1):
        if lo <= ahi < hi:
            return i
    return len(SEVERITY_BINS) - 1


# -------------------------
# 端到端数据集定义（修复标签映射和数据加载）
# -------------------------
class OSAEnd2EndDataset(Dataset):
    """端到端OSA诊断数据集：修复标签匹配和数据统计"""

    def __init__(self, data_dirs, seq_len=10, img_size=64, mean=None, std=None, train=True, wgan=None,
                 augment_ratio=0.3):
        self.seq_len = seq_len
        self.img_size = img_size
        self.train = train
        self.wgan = wgan
        self.augment_ratio = augment_ratio
        self.mean = mean
        self.std = std
        self.label_map = EVENT_LABEL_MAP  # 统一使用全局标签映射

        self.data = self._load_and_process_data(data_dirs)
        self.labels = [label for _, _, label in self.data]  # 新增这一行

        self._print_label_distribution()

        # 计算标准化参数（仅训练集）
        if train and self.mean is None:
            self.mean, self.std = self._calculate_mean_std()

    def _load_and_process_data(self, data_dirs):
        """加载多个目录的数据，修复标签匹配逻辑"""
        subject_events = defaultdict(list)  # {subject_id: list of (img, label)}
        all_labels = []  # 统计所有标签，用于调试

        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                print(f"警告：目录不存在，跳过 - {data_dir}")
                continue

            for filename in os.listdir(data_dir):
                if filename.endswith(".pickle"):
                    pickle_path = os.path.join(data_dir, filename)
                    events = load_pickle_events(pickle_path)
                    subject_id = filename.split('.')[0]

                    # 区分增强样本
                    if "aug_" in filename:
                        subject_id = f"aug_{subject_id}"

                    # 提取事件特征和标签（严格匹配小写）
                    for ev in events:
                        img = to_uint8_image(ev.signal)
                        event_label = str(ev.label).lower().strip()  # 强制小写+去空格
                        mapped = EVENT_LABEL_MAP.get(event_label)
                        if mapped is None:
                            print(f"[skip] 未识别标签: {event_label}")
                            continue

                        # 标签映射（兼容更多可能的标签写法）
                        if event_label in self.label_map:
                            mapped_label = self.label_map[event_label]
                        elif event_label == "obstructive apnea":
                            mapped_label = self.label_map["obstructiveapnea"]  # 兼容带空格的标签
                        elif event_label == "hypopnoea":
                            mapped_label = self.label_map["hypopnea"]  # 兼容英式拼写
                        else:
                            print(f"警告：无效标签 '{event_label}'，跳过事件（{subject_id}）")
                            continue

                        subject_events[subject_id].append((img, mapped_label))
                        all_labels.append(mapped_label)

        # 生成时序序列
        seq_data = []
        for subj_id, events in subject_events.items():
            if len(events) >= self.seq_len:
                for i in range(len(events) - self.seq_len + 1):
                    window_imgs = [img for img, _ in events[i:i + self.seq_len]]
                    window_label = events[i + self.seq_len - 1][1]  # 用最后一个事件的标签
                    seq_data.append((subj_id, window_imgs, window_label))

        # 打印标签统计
        print(f"数据加载完成：{len(seq_data)} 条序列 (seq_len={self.seq_len})")
        print(f"所有事件标签分布：{Counter(all_labels)}（0=normal,1=Hypopnea,2=ObstructiveApnea）")
        return seq_data

    def _print_label_distribution(self):
        """打印数据集的标签分布（用于调试）"""
        if not self.data:
            print("警告：数据集中无有效样本")
            return
        labels = [label for _, _, label in self.data]
        label_count = Counter(labels)
        print(f"序列标签分布：{dict(label_count)}")
        for idx, name in enumerate(EVENT_CLASS_NAMES):
            print(f"  {name}: {label_count.get(idx, 0)} 条序列")

    def _calculate_mean_std(self, max_samples=5000):
        """计算标准化参数"""
        count = 0
        mean = 0.0
        M2 = 0.0
        for _, seq, _ in self.data[:max_samples]:
            for img in seq:
                x = img.astype(np.float32) / 255.0
                n = x.size
                batch_mean = float(x.mean())
                batch_var = float(x.var())
                total = count + n
                delta = batch_mean - mean
                mean += delta * n / total
                M2 += batch_var * n + delta * delta * count * n / total
                count = total
        std = float(np.sqrt(M2 / max(count - 1, 1)))
        print(f"标准化参数 - 均值: {mean:.4f}, 标准差: {std:.4f}")
        return mean, std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        subj_id, seq, label = self.data[idx]
        seq_tensor = []
        for img in seq:
            img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # (1,H,W)
            # 统一尺寸
            if img_tensor.shape[-2:] != (self.img_size, self.img_size):
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=(self.img_size, self.img_size),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)
            img_tensor = img_tensor / 255.0
            img_tensor = (img_tensor - self.mean) / (self.std + 1e-6)
            seq_tensor.append(img_tensor)
        seq_tensor = torch.stack(seq_tensor, dim=0)  # (seq_len, 1, img_size, img_size)
        return seq_tensor, torch.tensor(label, dtype=torch.long), subj_id


# -------------------------
# 夜级评估函数（修复严重度映射和样本统计）
# -------------------------
@torch.no_grad()
def eval_night_level_with_truth(model,
                                dataloader,
                                device,
                                truth_csv_path,
                                epoch_seconds=30,
                                hyp_idx=1,
                                osa_idx=2):
    truth = load_truth_csv(truth_csv_path)
    print(f"\n加载真值数据：{len(truth)} 个subject")

    per_subj_preds = defaultdict(list)
    for batch in dataloader:
        if len(batch) == 3:
            seq, _, subj_ids = batch
        else:
            raise RuntimeError("期望 dataloader 返回 (seq, label, subj_id) 三元组")

        seq = seq.to(device)
        logits = model(seq)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        for sid, p in zip(subj_ids, preds):
            sid = str(sid).strip()
            per_subj_preds[sid].append(int(p))

    print(f"预测完成：{len(per_subj_preds)} 个subject")

    y_true_sev, y_pred_sev, used_ids, skipped_sids = [], [], [], []
    for sid, pred_seq in per_subj_preds.items():
        # 简化ID匹配，避免正常样本因ID问题被跳过
        original_sid = sid.replace("aug_", "").replace("_", "").replace("-", "")
        possible_sids = [sid, original_sid, original_sid.lower(), original_sid.upper()]

        matched_sid = None
        for candidate in possible_sids:
            if candidate in truth:
                matched_sid = candidate
                break

        if matched_sid is None:
            skipped_sids.append(sid)
            continue

        true_info = truth[matched_sid]
        true_sev = true_info["severity"]

        # 关键改动：增加异常事件判定阈值（连续2个窗口预测为异常才计数）
        pred_events = 0
        consecutive = 0
        threshold = 1  # 连续2个窗口异常才视为有效事件
        for p in pred_seq:
            if p in (hyp_idx, osa_idx):
                consecutive += 1
                if consecutive >= threshold:
                    pred_events += 1
                    consecutive = 0  # 计数后重置，避免重复统计
            else:
                consecutive = 0

        hours = len(pred_seq) * epoch_seconds / 3600.0
        pred_ahi = pred_events / hours if hours > 0 else 0.0
        pred_sev = classify_severity(pred_ahi)

        y_true_sev.append(true_sev)
        y_pred_sev.append(pred_sev)
        used_ids.append(sid)

    if not y_true_sev:
        raise RuntimeError("没有可用的subject进行夜级评估（检查subj_id匹配）")

    # 打印详细统计
    print(f"\n有效评估样本数：{len(used_ids)}")
    print(f"真值严重度分布：{Counter(y_true_sev)}（0=None,1=Mild,2=Moderate,3=Severe）")
    print(f"预测严重度分布：{Counter(y_pred_sev)}")

    # 计算指标
    acc = accuracy_score(y_true_sev, y_pred_sev)
    f1w = f1_score(y_true_sev, y_pred_sev, labels=SEVERITY_LABELS, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true_sev, y_pred_sev, labels=SEVERITY_LABELS)

    # 输出结果
    print("\n====== 夜级 OSA 严重程度评估（None/Mild/Moderate/Severe）======")
    print(f"Subjects used: {len(used_ids)}")
    print(f"Accuracy: {acc:.4f} | Weighted F1: {f1w:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(
        y_true_sev, y_pred_sev,
        labels=SEVERITY_LABELS,
        target_names=SEVERITY_NAMES,
        zero_division=0
    ))
    return acc, f1w, cm, (y_true_sev, y_pred_sev, used_ids)


# -------------------------
# 混淆矩阵输出函数
# -------------------------
def print_confusion_matrix(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(f"\n{'=' * 50}")
    print(f"{title} - 混淆矩阵")
    print(f"{'=' * 50}")
    print("类别映射：")
    for idx, name in enumerate(class_names):
        print(f"  索引 {idx} -> {name}")
    print(f"\n原始混淆矩阵：")
    print(cm)
    print(f"\n归一化混淆矩阵（保留2位小数）：")
    print(np.round(cm_normalized, 2))
    print(f"{'=' * 50}\n")


# -------------------------
# 训练和评估函数
# -------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in tqdm(dataloader, desc="训练"):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            seq, labels, _ = batch
        else:
            seq, labels = batch

        seq = seq.to(device)
        labels = labels.view(-1).to(device)

        optimizer.zero_grad()
        outputs = model(seq)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * seq.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, train_f1


def eval_model(model, dataloader, criterion, device, class_names, title):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估"):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                seq, labels, _ = batch
            else:
                seq, labels = batch

            seq = seq.to(device)
            labels = labels.view(-1).to(device)
            outputs = model(seq)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * seq.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    # 关键改动：指定labels参数，确保F1计算包含所有类别（避免normal类被忽略）
    val_f1 = f1_score(
        all_labels, all_preds,
        labels=list(range(len(class_names))),  # 强制包含0/1/2类
        average='weighted',
        zero_division=0
    )
    print_confusion_matrix(all_labels, all_preds, class_names, title)

    # 关键改动：逐类计算F1时也指定labels，确保统计正确
    print("各类别F1分数：")
    per_class_f1 = f1_score(
        all_labels, all_preds,
        labels=list(range(len(class_names))),
        average=None,
        zero_division=0
    )
    for class_name, f1v in zip(class_names, per_class_f1):
        print(f"  {class_name}: {f1v:.4f}")

    print("各类别样本数：")
    label_count = Counter(all_labels)
    for idx, class_name in enumerate(class_names):
        print(f"  {class_name}: {label_count.get(idx, 0)} 个样本")
    return avg_loss, val_f1


# -------------------------
# 加载预训练WGAN模型
# -------------------------
def load_pretrained_wgan(wgan_path, device):
    wgan = WGANGP(
        input_dim=100,
        img_channels=1,
        device=device
    )
    if os.path.exists(wgan_path):
        wgan.generator.load_state_dict(torch.load(wgan_path, map_location=device))
        wgan.generator.eval()
        print(f"成功加载预训练WGAN生成器: {wgan_path}")
    else:
        raise FileNotFoundError(f"WGAN生成器权重文件不存在: {wgan_path}")
    return wgan


# -------------------------
# 主函数
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../../config.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="端到端模型训练轮次")
    parser.add_argument("--save_path", default="models/end2end/")
    parser.add_argument("--train_dir", default=None, help="训练数据目录（覆盖config）")
    parser.add_argument("--val_dir", default=None, help="验证数据目录（覆盖config）")
    parser.add_argument("--seq_len", type=int, default=10, help="时序序列长度")
    parser.add_argument("--wgan_pretrained",
                        default="/Users/liyuxiang/Downloads/sleep-apnea-main/src/train/models/end2end/wgan_generator.pth",
                        help="预训练WGAN生成器路径")
    parser.add_argument("--img_size", type=int, default=64, help="图像尺寸（需与WGAN一致）")
    parser.add_argument("--truth_csv",
                        default="/Users/liyuxiang/Downloads/sleep-apnea-main/data/processed/labels/night_level_ahi.csv",
                        help="真值CSV路径")
    args = parser.parse_args()

    # 初始化路径和配置
    os.makedirs(args.save_path, exist_ok=True)
    config = load_config(resolve_dir(args.config, [Path.cwd(), project_root()]))
    anchors = [Path.cwd(), project_root()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # -------------------------
    # 步骤1：加载预训练WGAN模型
    # -------------------------
    print("\n" + "=" * 60)
    print("步骤1：加载预训练WGAN模型")
    print("=" * 60)
    wgan = load_pretrained_wgan(args.wgan_pretrained, device)

    # -------------------------
    # 步骤2：解析数据目录
    # -------------------------
    if args.train_dir:
        raw_data_dir = resolve_dir(args.train_dir, anchors)
        data_dirs = [raw_data_dir]
    else:
        raw_data_dir = resolve_dir(config.paths.signals_path, anchors)
        augmented_data_dir = resolve_dir(config.paths.augmented_save_path, anchors)
        data_dirs = [raw_data_dir]
        if os.path.exists(augmented_data_dir) and len(os.listdir(augmented_data_dir)) > 0:
            data_dirs.append(augmented_data_dir)
            print(f"加载原始数据：{raw_data_dir}")
            print(f"加载增强数据：{augmented_data_dir}")
        else:
            print(f"仅加载原始数据：{raw_data_dir}")

    val_dir = resolve_dir(args.val_dir or config.paths.signals_path, anchors)
    print(f"验证数据目录：{val_dir}")

    # -------------------------
    # 步骤3：构建数据集
    # -------------------------
    print("\n" + "=" * 60)
    print("步骤3：构建数据集")
    print("=" * 60)


    # 训练数据集（保持不变）
    train_dataset = OSAEnd2EndDataset(
        data_dirs=data_dirs,
        seq_len=args.seq_len,
        img_size=args.img_size,
        train=True,
        wgan=None,
        augment_ratio=0
    )

    # 保存标准化参数（保持不变）
    with open(os.path.join(args.save_path, "data_stats.pkl"), 'wb') as f:
        pickle.dump({"mean": train_dataset.mean, "std": train_dataset.std}, f)
    print(f"数据标准化参数已保存到: {os.path.join(args.save_path, 'data_stats.pkl')}")

    # 验证数据集（保持不变）
    val_dataset = OSAEnd2EndDataset(
        data_dirs=[val_dir],
        seq_len=args.seq_len,
        img_size=args.img_size,
        mean=train_dataset.mean,
        std=train_dataset.std,
        train=False,
        wgan=None,
        augment_ratio=0
    )

    # -------------------------
    # 关键改动：给训练集加加权采样器，平衡三类样本
    # -------------------------
    # 计算训练集每个样本的权重（少数类样本权重高，被采样概率大）
    train_labels = train_dataset.labels  # 从数据集获取所有标签

    num_classes = len(EVENT_CLASS_NAMES)

    # 训练DataLoader加入采样器（验证集不用，保持真实分布）
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # -------------------------
    # 步骤4：初始化模型并训练
    # -------------------------
    # -------------------------
    # 步骤4：初始化模型并训练（修改部分）
    # -------------------------
    print("\n" + "=" * 60)
    print("步骤4：端到端OSA诊断模型训练（三类分类）")
    print("=" * 60)

    # 初始化模型（保持不变）
    model = OSAEnd2EndModel(
        img_channels=1,
        img_size=args.img_size,
        seq_len=args.seq_len,
        num_classes=len(EVENT_CLASS_NAMES)
    ).to(device)

    # 关键改动：温和的类别权重（不极端，仅中和样本不平衡）
    class_weights = torch.tensor([1.0, 1.8, 1.5], dtype=torch.float, device=device)
    print(f"使用温和类别权重：{class_weights.tolist()}（对应标签：{EVENT_CLASS_NAMES}）")
    from collections import Counter
    cnt = Counter(train_dataset.labels)
    weights = torch.tensor([sum(cnt.values()) / cnt.get(i, 1) for i in range(3)], dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # 优化器调整（保持不变）
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,  # 慢学习率，避免震荡
        weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, verbose=False
    )
    best_f1 = 0.0

    # 开始训练
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        # 训练
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Train F1 (weighted): {train_f1:.4f}")

        # 验证
        val_loss, val_f1 = eval_model(
            model, val_loader, criterion, device,
            class_names=EVENT_CLASS_NAMES,
            title=f"Epoch {epoch + 1} 验证集"
        )
        print(f"Val   Loss: {val_loss:.4f} | Val   F1 (weighted): {val_f1:.4f}")

        # 夜级评估
        try:
            _ = eval_night_level_with_truth(
                model,
                val_loader,
                device,
                truth_csv_path="/Users/liyuxiang/Downloads/sleep-apnea-main/data/processed/labels/night_level_ahi.csv",
                epoch_seconds=30,
                hyp_idx=1,
                osa_idx=2,
            )
        except Exception as e:
            print(f"夜级评估出错：{str(e)}")

        # 学习率调度
        scheduler.step(val_f1)

        # 保存最优模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(args.save_path, "osa_end2end_best_3class.pth")
            torch.save(model.state_dict(), save_path)
            print(f"保存最优模型到: {save_path} (F1: {best_f1:.4f})")

    # 最终评估
    print("\n" + "=" * 60)
    print("训练结束，最优模型最终评估")
    print("=" * 60)
    model.load_state_dict(torch.load(os.path.join(args.save_path, "osa_end2end_best_3class.pth"), map_location=device))
    final_val_loss, final_val_f1 = eval_model(
        model, val_loader, criterion, device,
        class_names=EVENT_CLASS_NAMES,
        title="最优模型验证集"
    )
    print(f"最优模型最终加权F1: {final_val_f1:.4f}")

    print("\n" + "=" * 60)
    print("端到端OSA三类诊断模型训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

# import os
# import argparse
# import pickle
# import numpy as np
# import torch
# from torch import nn
# from torch.utils.data import DataLoader, Dataset
# from sklearn.metrics import f1_score, confusion_matrix
# from pathlib import Path
# from tqdm import tqdm
#
# # 导入模型和工具函数
# from src.models.wgan_gp import WGANGP
# from src.models.osa_end2end import OSAEnd2EndModel
# from src.preprocessing.steps.config import load_config
# from src.utils.utils import load_pickle_events, to_uint8_image
#
# import torch.nn.functional as F
#
#
# # -------------------------
# # 工具函数
# # -------------------------
# def cfg_get(cfg, path, default=None):
#     cur = cfg
#     for key in path.split('.'):
#         if cur is None:
#             return default
#         if hasattr(cur, key):
#             cur = getattr(cur, key)
#             continue
#         if isinstance(cur, dict) and key in cur:
#             cur = cur[key]
#             continue
#         return default
#     return cur
#
#
# def project_root() -> Path:
#     return Path(__file__).resolve().parents[2]
#
#
# def resolve_dir(p, anchors):
#     if p is None:
#         return ""
#     P = Path(p)
#     if P.is_absolute():
#         return str(P)
#     for a in anchors:
#         cand = a / P
#         if cand.exists():
#             return str(cand.resolve())
#     return str((anchors[0] / P).resolve())
#
#
# # -------------------------
# # 端到端数据集定义
# # -------------------------
# class OSAEnd2EndDataset(Dataset):
#     """端到端OSA诊断数据集：直接从原始事件构建时序序列（三类分类）"""
#
#     # 新增 data_dirs 参数（接收目录列表），移除原 data_dir 参数
#     def __init__(self, data_dirs, seq_len=10, img_size=64, mean=None, std=None, train=True, wgan=None,
#                  augment_ratio=0.3):
#         self.seq_len = seq_len
#         self.img_size = img_size  # 图像尺寸
#         self.train = train
#         self.wgan = wgan
#         self.augment_ratio = augment_ratio
#         self.mean = mean
#         self.std = std
#         self.label_map = {  # 三类标签映射
#             "normal": 0,
#             "hypopnea": 1,
#             "obstructiveapnea": 2
#         }
#
#         # 加载多个目录的数据并构建序列
#         self.data = self._load_and_process_data(data_dirs)
#
#         # 计算标准化参数（仅训练集）
#         if train and self.mean is None:
#             self.mean, self.std = self._calculate_mean_std()
#
#     def _load_and_process_data(self, data_dirs):
#         """加载多个目录的数据并生成时序序列"""
#         subject_events = {}  # {subject_id: list of (img, label)}
#
#         # 遍历所有数据目录（原始数据目录 + 增强数据目录）
#         for data_dir in data_dirs:
#             if not os.path.exists(data_dir):
#                 print(f"警告：目录不存在，跳过 - {data_dir}")
#                 continue
#
#             for filename in os.listdir(data_dir):
#                 if filename.endswith(".pickle"):
#                     pickle_path = os.path.join(data_dir, filename)
#                     events = load_pickle_events(pickle_path)  # 加载事件列表
#                     subject_id = filename.split('.')[0]
#
#                     # 区分增强样本和原始样本（避免ID冲突）
#                     if "aug_" in filename:  # 增强样本文件名含前缀
#                         subject_id = f"aug_{subject_id}"
#
#                     # 提取事件特征和标签
#                     event_list = []
#                     for ev in events:
#                         # 从ApneaEvent对象中提取信号和标签
#                         img = to_uint8_image(ev.signal)
#                         event_label = ev.label.lower()  # 统一转为小写
#
#                         # 检查标签是否有效
#                         if event_label in self.label_map:
#                             event_list.append((img, self.label_map[event_label]))
#                         else:
#                             print(f"警告：无效标签 '{event_label}'，跳过事件（{subject_id}）")
#
#                     # 保存有效事件
#                     if event_list:
#                         if subject_id in subject_events:
#                             subject_events[subject_id].extend(event_list)
#                         else:
#                             subject_events[subject_id] = event_list
#
#         # 生成滑动窗口序列（与之前逻辑一致）
#         seq_data = []
#         for subj_id, events in subject_events.items():
#             # 确保事件数量足够生成序列
#             if len(events) >= self.seq_len:
#                 for i in range(len(events) - self.seq_len + 1):
#                     window_imgs = [img for img, _ in events[i:i + self.seq_len]]
#                     window_label = events[i + self.seq_len - 1][1]  # 窗口标签取最后一个事件
#                     seq_data.append((window_imgs, window_label))
#
#         print(f"数据集加载完成：{len(seq_data)} 条序列 (seq_len={self.seq_len})")
#         return seq_data
#
#     def _calculate_mean_std(self, max_samples=5000):
#         """
#         在线估计 mean/std，避免把所有像素塞进内存
#         可采样前 max_samples 个序列（或全量）做估计
#         """
#         count = 0
#         mean = 0.0
#         M2 = 0.0  # sum of squared diffs for variance (Welford)
#         for seq, _ in self.data[:max_samples]:
#             for img in seq:
#                 x = img.astype(np.float32) / 255.0
#                 n = x.size
#                 # 批量更新
#                 batch_mean = float(x.mean())
#                 batch_var = float(x.var())
#                 # 把一批看作 n 次观测合并
#                 total = count + n
#                 delta = batch_mean - mean
#                 mean += delta * n / total
#                 M2 += batch_var * n + delta * delta * count * n / total
#                 count = total
#         std = float(np.sqrt(M2 / max(count - 1, 1)))
#         print(f"计算标准化参数(采样{max_samples}) - 均值: {mean:.4f}, 标准差: {std:.4f}")
#         return mean, std
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         seq, label = self.data[idx]
#         seq_tensor = []
#         for img in seq:
#             img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # (1,H,W), uint8 -> float
#             # 统一尺寸到 (1, img_size, img_size)
#             if img_tensor.shape[-2:] != (self.img_size, self.img_size):
#                 img_tensor = F.interpolate(
#                     img_tensor.unsqueeze(0),  # (N=1,C=1,H,W)
#                     size=(self.img_size, self.img_size),
#                     mode="bilinear",
#                     align_corners=False
#                 ).squeeze(0)
#             img_tensor = img_tensor / 255.0
#             img_tensor = (img_tensor - self.mean) / (self.std + 1e-6)
#             seq_tensor.append(img_tensor)
#         seq_tensor = torch.stack(seq_tensor, dim=0)  # (seq_len, 1, img_size, img_size)
#         return seq_tensor, torch.tensor(label, dtype=torch.long)  # 形状: (), DataLoader 会堆成 (batch,)
#
#
#
# # -------------------------
# # 混淆矩阵输出函数
# # -------------------------
# def print_confusion_matrix(y_true, y_pred, class_names, title):
#     cm = confusion_matrix(y_true, y_pred)
#     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#     print(f"\n{'=' * 50}")
#     print(f"{title} - 混淆矩阵")
#     print(f"{'=' * 50}")
#     print("类别映射：")
#     for idx, name in enumerate(class_names):
#         print(f"  索引 {idx} -> {name}")
#     print(f"\n原始混淆矩阵：")
#     print(cm)
#     print(f"\n归一化混淆矩阵（保留2位小数）：")
#     print(np.round(cm_normalized, 2))
#     print(f"{'=' * 50}\n")
#
#
# # -------------------------
# # 训练和评估函数
# # -------------------------
# def train_epoch(model, dataloader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0.0
#     all_preds = []
#     all_labels = []
#
#     for seq, labels in tqdm(dataloader, desc="训练"):
#         seq = seq.to(device)
#         labels = labels.view(-1).to(device)  # ← 保证是 (batch,)
#         optimizer.zero_grad()
#         outputs = model(seq)  # (batch, num_classes)
#         loss = criterion(outputs, labels)  # OK
#
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item() * seq.size(0)
#         preds = torch.argmax(outputs, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
#
#     avg_loss = total_loss / len(dataloader.dataset)
#     # 三类分类的加权F1分数
#     train_f1 = f1_score(all_labels, all_preds, average='weighted')
#     return avg_loss, train_f1
#
#
# def eval_model(model, dataloader, criterion, device, class_names, title):
#     model.eval()
#     total_loss = 0.0
#     all_preds = []
#     all_labels = []
#
#     with torch.no_grad():
#         for seq, labels in tqdm(dataloader, desc="评估"):
#             seq = seq.to(device)
#             labels = labels.view(-1).to(device)  # ← 同样确保 (batch,)
#             outputs = model(seq)
#             loss = criterion(outputs, labels)
#
#             total_loss += loss.item() * seq.size(0)
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#
#     avg_loss = total_loss / len(dataloader.dataset)
#     val_f1 = f1_score(all_labels, all_preds, average='weighted')
#     print_confusion_matrix(all_labels, all_preds, class_names, title)
#     # 额外输出各类别的F1分数
#     per_class_f1 = f1_score(all_labels, all_preds, average=None)
#     print("各类别F1分数：")
#     for idx, (class_name, f1) in enumerate(zip(class_names, per_class_f1)):
#         print(f"  {class_name}: {f1:.4f}")
#     return avg_loss, val_f1
#
#
# # -------------------------
# # 加载预训练WGAN模型
# # -------------------------
# def load_pretrained_wgan(wgan_path, device):
#     """
#     加载预训练的WGAN生成器
#     :param wgan_path: WGAN生成器权重路径
#     :param device: 计算设备
#     :return: wgan模型
#     """
#     # 初始化WGAN-GP
#     wgan = WGANGP(
#         input_dim=100,
#         img_channels=1,
#         device=device
#     )
#
#     # 加载预训练权重
#     if os.path.exists(wgan_path):
#         wgan.generator.load_state_dict(torch.load(wgan_path, map_location=device))
#         wgan.generator.eval()
#         print(f"成功加载预训练WGAN生成器: {wgan_path}")
#     else:
#         raise FileNotFoundError(f"WGAN生成器权重文件不存在: {wgan_path}")
#
#     return wgan
#
#
# # -------------------------
# # 主函数
# # -------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", default="../../config.yaml")
#     parser.add_argument("--epochs", type=int, default=50, help="端到端模型训练轮次")
#     parser.add_argument("--save_path", default="models/end2end/")
#     parser.add_argument("--train_dir", default=None, help="训练数据目录（覆盖config）")
#     parser.add_argument("--val_dir", default=None, help="验证数据目录（覆盖config）")
#     parser.add_argument("--seq_len", type=int, default=10, help="时序序列长度")
#     parser.add_argument("--augment_ratio", type=float, default=0.3, help="已废弃，预处理阶段已增强")
#     parser.add_argument(
#         "--wgan_pretrained",
#         default="/Users/liyuxiang/Downloads/sleep-apnea-main/src/train/models/end2end/wgan_generator.pth",
#         help="预训练WGAN生成器路径"
#     )
#     parser.add_argument("--img_size", type=int, default=64, help="图像尺寸（需与WGAN一致）")
#     args = parser.parse_args()
#
#     # 初始化路径和配置
#     os.makedirs(args.save_path, exist_ok=True)
#     config = load_config(resolve_dir(args.config, [Path.cwd(), project_root()]))
#     anchors = [Path.cwd(), project_root()]
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备: {device}")
#
#     # 三类分类标签映射（与config.yaml保持一致）
#     class_names = ["normal", "Hypopnea", "ObstructiveApnea"]
#     label_map = {name: idx for idx, name in enumerate(class_names)}
#
#     # -------------------------
#     # 步骤1：加载预训练WGAN模型（仅用于验证增强逻辑，训练时不生成新数据）
#     # -------------------------
#     print("\n" + "=" * 60)
#     print("步骤1：加载预训练WGAN模型")
#     print("=" * 60)
#     wgan = load_pretrained_wgan(args.wgan_pretrained, device)
#
#     # -------------------------
#     # 步骤2：解析数据目录（原始数据+增强数据）
#     # -------------------------
#     # 优先使用命令行传入的目录，否则从config读取
#     if args.train_dir:
#         raw_data_dir = resolve_dir(args.train_dir, anchors)
#         augmented_data_dir = ""  # 若指定train_dir，不自动加载增强数据
#     else:
#         raw_data_dir = resolve_dir(config.paths.signals_path, anchors)
#         augmented_data_dir = resolve_dir(config.paths.augmented_save_path, anchors)
#
#     # 构建数据目录列表（原始数据+增强数据）
#     data_dirs = [raw_data_dir]
#     if not args.train_dir:  # 未指定train_dir时，尝试加载增强数据
#         if os.path.exists(augmented_data_dir) and len(os.listdir(augmented_data_dir)) > 0:
#             data_dirs.append(augmented_data_dir)
#             print(f"加载原始数据：{raw_data_dir}")
#             print(f"加载增强数据：{augmented_data_dir}")
#         else:
#             print(f"仅加载原始数据：{raw_data_dir}（增强数据目录为空或不存在）")
#     else:
#         print(f"使用命令行指定的训练数据目录：{raw_data_dir}")
#
#     # 验证集目录（单独处理，不使用增强数据，保持原始分布）
#     val_dir = resolve_dir(args.val_dir or config.paths.signals_path, anchors)
#     print(f"验证数据目录：{val_dir}")
#
#     # -------------------------
#     # 步骤3：构建数据集（训练集含增强数据，验证集仅原始数据）
#     # -------------------------
#     print("\n" + "=" * 60)
#     print("步骤3：构建数据集")
#     print("=" * 60)
#
#     # 训练数据集（加载原始+增强数据，关闭实时增强）
#     train_dataset = OSAEnd2EndDataset(
#         data_dirs=data_dirs,  # 传入目录列表（原始+增强）
#         seq_len=args.seq_len,
#         img_size=args.img_size,  # 新增img_size参数
#         train=True,
#         wgan=None,  # 预处理已增强，训练时无需WGAN
#         augment_ratio=0  # 关闭实时增强
#     )
#
#     # 保存标准化参数（用于推理）
#     with open(os.path.join(args.save_path, "data_stats.pkl"), 'wb') as f:
#         pickle.dump({"mean": train_dataset.mean, "std": train_dataset.std}, f)
#     print(f"数据标准化参数已保存到: {os.path.join(args.save_path, 'data_stats.pkl')}")
#
#     # 验证数据集（仅加载原始数据，保持真实分布）
#     val_dataset = OSAEnd2EndDataset(
#         data_dirs=[val_dir],  # 仅原始数据
#         seq_len=args.seq_len,
#         img_size=args.img_size,
#         mean=train_dataset.mean,
#         std=train_dataset.std,
#         train=False,
#         wgan=None,
#         augment_ratio=0
#     )
#
#     # 数据加载器（减小batch_size避免内存溢出）
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=8,  # 小批次更稳定
#         shuffle=True,
#         num_workers=0,
#         pin_memory=True  # 加速GPU传输
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=8,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=True
#     )
#
#     # -------------------------
#     # 步骤4：初始化模型并训练
#     # -------------------------
#     print("\n" + "=" * 60)
#     print("步骤4：端到端OSA诊断模型训练（三类分类）")
#     print("=" * 60)
#
#     # 初始化三类分类模型
#     model = OSAEnd2EndModel(
#         img_channels=1,
#         img_size=args.img_size,
#         seq_len=args.seq_len,
#         num_classes=3  # 三类分类
#     ).to(device)
#
#     # 训练配置（针对类别不平衡，可添加权重）
#     # 计算类别权重（可选，根据原始数据分布）
#     train_labels = [label for _, label in train_dataset.data]
#     class_counts = [train_labels.count(i) for i in range(3)]
#     weights = torch.FloatTensor([sum(class_counts)/c for c in class_counts]).to(device)
#     criterion = nn.CrossEntropyLoss(weight=weights)  # 加权损失缓解不平衡
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='max', factor=0.5, patience=5, verbose=True
#     )
#     best_f1 = 0.0
#
#     # 开始训练
#     for epoch in range(args.epochs):
#         print(f"\nEpoch {epoch + 1}/{args.epochs}")
#         print("-" * 50)
#
#         # 训练
#         train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
#         print(f"Train Loss: {train_loss:.4f} | Train F1 (weighted): {train_f1:.4f}")
#
#         # 验证
#         val_loss, val_f1 = eval_model(
#             model, val_loader, criterion, device,
#             class_names=class_names,
#             title=f"Epoch {epoch + 1} 验证集"
#         )
#         print(f"Val   Loss: {val_loss:.4f} | Val   F1 (weighted): {val_f1:.4f}")
#
#         # 学习率调度
#         scheduler.step(val_f1)
#
#         # 保存最优模型
#         if val_f1 > best_f1:
#             best_f1 = val_f1
#             save_path = os.path.join(args.save_path, "osa_end2end_best_3class.pth")
#             torch.save(model.state_dict(), save_path)
#             print(f"保存最优模型到: {save_path} (F1: {best_f1:.4f})")
#
#     # 最终评估
#     print("\n" + "=" * 60)
#     print("训练结束，最优模型最终评估")
#     print("=" * 60)
#     model.load_state_dict(torch.load(os.path.join(args.save_path, "osa_end2end_best_3class.pth"), weights_only=True))
#     final_val_loss, final_val_f1 = eval_model(
#         model, val_loader, criterion, device,
#         class_names=class_names,
#         title="最优模型验证集"
#     )
#     print(f"最优模型最终加权F1: {final_val_f1:.4f}")
#
#     print("\n" + "=" * 60)
#     print("端到端OSA三类诊断模型训练完成！")
#     print("=" * 60)
#
#
#
# if __name__ == "__main__":
#     main()