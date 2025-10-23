import os
import argparse
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path
from tqdm import tqdm

# 导入模型和工具函数
from src.models.wgan_gp import WGANGP
from src.models.osa_end2end import OSAEnd2EndModel
from src.preprocessing.steps.config import load_config
from src.utils.utils import load_pickle_events, to_uint8_image

import torch.nn.functional as F


# -------------------------
# 工具函数
# -------------------------
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


# -------------------------
# 端到端数据集定义
# -------------------------
class OSAEnd2EndDataset(Dataset):
    """端到端OSA诊断数据集：直接从原始事件构建时序序列"""

    def __init__(self, data_dir, seq_len=10, mean=None, std=None, train=True, wgan=None, augment_ratio=0.3):
        self.seq_len = seq_len
        self.train = train
        self.wgan = wgan
        self.augment_ratio = augment_ratio
        self.mean = mean
        self.std = std

        # 加载数据并构建序列
        self.data = self._load_and_process_data(data_dir)

        # 计算标准化参数（仅训练集）
        if train and self.mean is None:
            self.mean, self.std = self._calculate_mean_std()

    def _load_and_process_data(self, data_dir):
        """按受试者加载事件数据并生成时序序列"""
        subject_events = {}  # {subject_id: (event_list, osa_label)}

        for filename in os.listdir(data_dir):
            if filename.endswith(".pickle"):
                pickle_path = os.path.join(data_dir, filename)
                events = load_pickle_events(pickle_path)
                subject_id = filename.split('.')[0]

                # 提取OSA标签（根据实际标签规则修改）
                osa_label = 1 if "apnea" in subject_id.lower() else 0

                # 处理事件的梅尔频谱特征
                event_list = []
                for ev in events:
                    img = to_uint8_image(ev.signal)  # 转换为uint8图像
                    event_list.append(img)

                if len(event_list) >= self.seq_len:
                    subject_events[subject_id] = (event_list, osa_label)

        # 生成滑动窗口序列
        seq_data = []
        for subj_id, (events, label) in subject_events.items():
            # 按时间排序（假设事件已按时间排序，若未排序需补充排序逻辑）
            for i in range(len(events) - self.seq_len + 1):
                window = events[i:i + self.seq_len]
                seq_data.append((window, label))

        # 训练集数据增强（使用WGAN生成虚拟数据）
        if self.train and self.wgan is not None and self.augment_ratio > 0:
            n_augment = int(len(seq_data) * self.augment_ratio)
            print(f"使用WGAN生成 {n_augment} 条增强数据...")
            fake_imgs, fake_labels = self.wgan.generate_data(n_augment)

            # 将单张生成图像构建为序列（重复填充至seq_len）
            for img, lbl in zip(fake_imgs, fake_labels):
                fake_seq = [img for _ in range(self.seq_len)]
                seq_data.append((fake_seq, lbl))

        print(f"数据集加载完成：{len(seq_data)} 条序列 (seq_len={self.seq_len})")
        return seq_data

    def _calculate_mean_std(self):
        """计算训练数据的标准化参数"""
        all_pixels = []
        for seq, _ in self.data:
            for img in seq:
                all_pixels.extend(img.flatten())
        mean = np.mean(all_pixels) / 255.0
        std = np.std(all_pixels) / 255.0
        print(f"计算标准化参数 - 均值: {mean:.4f}, 标准差: {std:.4f}")
        return mean, std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]

        # 转换为tensor并标准化
        seq_tensor = []
        for img in seq:
            img_tensor = torch.FloatTensor(img).unsqueeze(0)  # (1, H, W)
            img_tensor = img_tensor / 255.0  # 归一化到[0,1]
            img_tensor = (img_tensor - self.mean) / self.std  # 标准化
            seq_tensor.append(img_tensor)

        # 拼接为时序序列：(seq_len, 1, H, W)
        seq_tensor = torch.stack(seq_tensor, dim=0)
        return seq_tensor, torch.LongTensor([label])


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
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for seq, labels in tqdm(dataloader, desc="训练"):
        seq, labels = seq.to(device), labels.squeeze().to(device)
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
    train_f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, train_f1


def eval_model(model, dataloader, criterion, device, class_names, title):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for seq, labels in tqdm(dataloader, desc="评估"):
            seq, labels = seq.to(device), labels.squeeze().to(device)
            outputs = model(seq)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * seq.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    print_confusion_matrix(all_labels, all_preds, class_names, title)
    return avg_loss, val_f1


# -------------------------
# 主函数
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../../config.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="端到端模型训练轮次")
    parser.add_argument("--wgan_epochs", type=int, default=20, help="WGAN训练轮次")
    parser.add_argument("--save_path", default="models/end2end/")
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--seq_len", type=int, default=10, help="时序序列长度")
    parser.add_argument("--augment_ratio", type=float, default=0.3, help="WGAN数据增强比例")
    args = parser.parse_args()

    # 初始化路径
    os.makedirs(args.save_path, exist_ok=True)
    config = load_config(resolve_dir(args.config, [Path.cwd(), project_root()]))
    anchors = [Path.cwd(), project_root()]
    train_dir = resolve_dir(args.train_dir or cfg_get(config, 'paths.signals_path'), anchors)
    val_dir = resolve_dir(args.val_dir or cfg_get(config, 'paths.signals_path'), anchors)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # -------------------------
    # 步骤1：训练WGAN-GP数据增强模型
    # -------------------------
    print("\n" + "=" * 60)
    print("步骤1：训练WGAN-GP数据增强模型")
    print("=" * 60)

    # 加载原始图像数据用于WGAN训练
    def load_wgan_training_data(data_dir, max_samples=1000):
        imgs = []
        for filename in os.listdir(data_dir):
            if filename.endswith(".pickle") and len(imgs) < max_samples:
                pickle_path = os.path.join(data_dir, filename)
                events = load_pickle_events(pickle_path)
                for ev in events:
                    img = to_uint8_image(ev.signal)
                    imgs.append(img)
                    if len(imgs) >= max_samples:
                        break
        # 转换为tensor并归一化到[-1, 1]
        imgs = np.array(imgs)[:, np.newaxis, :, :]  # (N, 1, H, W)
        imgs = (imgs / 255.0) * 2 - 1  # 转换到[-1,1]范围
        return torch.FloatTensor(imgs)

    # 准备WGAN训练数据
    wgan_train_data = load_wgan_training_data(train_dir)
    print(f"WGAN训练数据量: {len(wgan_train_data)} 张图像")

    # 初始化WGAN-GP
    img_size = wgan_train_data.shape[2]  # 获取图像尺寸
    wgan = WGANGP(
        input_dim=100,
        img_channels=1,
        device=device
    )

    # 训练WGAN
    wgan_train_loader = DataLoader(
        wgan_train_data,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    for epoch in range(args.wgan_epochs):
        d_losses, g_losses = [], []
        for batch in wgan_train_loader:
            real_imgs = batch.to(device)
            d_loss, g_loss = wgan.train_step(real_imgs)
            d_losses.append(d_loss)
            g_losses.append(g_loss)

        avg_d_loss = np.mean(d_losses)
        avg_g_loss = np.mean(g_losses)
        print(f"WGAN Epoch {epoch + 1}/{args.wgan_epochs} | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

    # 保存WGAN生成器
    wgan_save_path = os.path.join(args.save_path, "wgan_generator.pth")
    torch.save(wgan.generator.state_dict(), wgan_save_path)
    print(f"WGAN生成器已保存到: {wgan_save_path}")

    # -------------------------
    # 步骤2：端到端OSA诊断模型训练
    # -------------------------
    print("\n" + "=" * 60)
    print("步骤2：端到端OSA诊断模型训练")
    print("=" * 60)

    # 构建数据集
    train_dataset = OSAEnd2EndDataset(
        train_dir,
        seq_len=args.seq_len,
        train=True,
        wgan=wgan,
        augment_ratio=args.augment_ratio
    )
    val_dataset = OSAEnd2EndDataset(
        val_dir,
        seq_len=args.seq_len,
        mean=train_dataset.mean,
        std=train_dataset.std,
        train=False
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    # 初始化创新诊断模型
    model = OSAEnd2EndModel(
        img_channels=1,
        img_size=img_size,
        seq_len=args.seq_len,
        num_classes=2
    ).to(device)

    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    class_names = ["Normal", "OSA"]
    best_f1 = 0.0

    # 开始训练
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        # 训练
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")

        # 验证
        val_loss, val_f1 = eval_model(
            model, val_loader, criterion, device,
            class_names=class_names,
            title=f"OSA诊断模型 - Epoch {epoch + 1} 验证集"
        )
        print(f"Val   Loss: {val_loss:.4f} | Val   F1: {val_f1:.4f}")

        # 学习率调度
        scheduler.step(val_f1)

        # 保存最优模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(args.save_path, "osa_end2end_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"保存最优模型到: {save_path} (F1: {best_f1:.4f})")

    # 最终评估
    print("\n" + "=" * 60)
    print("训练结束，最优模型最终评估")
    print("=" * 60)
    model.load_state_dict(torch.load(os.path.join(args.save_path, "osa_end2end_best.pth")))
    final_val_loss, final_val_f1 = eval_model(
        model, val_loader, criterion, device,
        class_names=class_names,
        title="OSA诊断模型 - 最优模型验证集"
    )
    print(f"最优模型最终F1: {final_val_f1:.4f}")

    print("\n" + "=" * 60)
    print("端到端OSA诊断模型训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()