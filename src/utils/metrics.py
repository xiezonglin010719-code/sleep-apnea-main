from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import torch
import torch.nn.functional as F
from typing import List, Dict, Any

import torch
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from src.federated.client import FederatedClient


def calculate_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """计算F1分数"""
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, save_path: str):
    """绘制混淆矩阵并保存"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> str:
    """生成分类报告"""
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)


def _resolve_labels_and_names(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], present_only: bool = False):
    """
    当 present_only=False：固定用全类别 [0..C-1] 输出（缺失类支持度为 0，zero_division=0）
    当 present_only=True ：只报告实际出现过的类别（报告更紧凑）
    """
    C = len(class_names)
    if present_only:
        present = sorted(set(map(int, np.unique(np.concatenate([y_true, y_pred])))))
        labels = [l for l in present if 0 <= l < C] or [0]
    else:
        labels = list(range(C))
    target_names = [class_names[i] for i in labels]
    return labels, target_names

def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], present_only: bool = False) -> str:
    labels, target_names = _resolve_labels_and_names(y_true, y_pred, class_names, present_only=present_only)
    return classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    )


@torch.no_grad()
def evaluate_classification(model, dataloader, device, class_names: List[str]) -> Dict[str, Any]:
    model.eval()
    all_true, all_preds, all_losses = [], [], []
    crit = torch.nn.CrossEntropyLoss()

    for X, y in dataloader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(X)
        # 兼容 dict/tuple 输出
        if isinstance(out, dict):
            logits = out.get("logits", None)
            if logits is None:
                # 退而求其次：取第一个 tensor
                for v in out.values():
                    if torch.is_tensor(v):
                        logits = v
                        break
        elif isinstance(out, (list, tuple)):
            logits = next((t for t in out if torch.is_tensor(t)), None)
        else:
            logits = out
        if logits is None:
            raise TypeError("Model forward must return logits tensor or dict with 'logits'.")

        # 若是 [N,T,C]，对 T 做均值池化
        if logits.dim() == 3:
            logits = logits.mean(dim=1)

        if y.dim() == 2 and y.size(1) > 1:
            # one-hot -> 索引
            y = torch.argmax(y, dim=1)
        else:
            y = y.view(-1)
        y = y.long()

        # dtype 对齐
        logits = logits.float()



        loss = crit(logits, y)
        all_losses.append(loss.item() * y.size(0))

        pred = torch.argmax(logits, dim=1)
        all_true.append(y.detach().cpu())
        all_preds.append(pred.detach().cpu())

    if not all_true:
        return {
            "loss": 0.0,
            "f1_macro": 0.0,
            "f1_micro": 0.0,
            "classification_report": "No samples.",
            "y_true": np.array([], dtype=int),
            "y_pred": np.array([], dtype=int),
            "confusion_matrix": np.zeros((len(class_names), len(class_names)), dtype=int),
        }


    y_true = torch.cat(all_true).numpy().astype(int)
    y_pred = torch.cat(all_preds).numpy().astype(int)

    # 计算 loss
    loss = float(np.sum(all_losses) / max(len(y_true), 1))

    # labels/target_names 对齐（固定全类别输出，缺失类支持度为 0）
    labels, target_names = _resolve_labels_and_names(y_true, y_pred, class_names, present_only=False)

    # F1（确保用同一套 labels）
    f1_macro = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    f1_micro = float(f1_score(y_true, y_pred, average="micro", labels=labels, zero_division=0))

    # 报告
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    )

    # 混淆矩阵（固定大小 C×C，便于绘制）
    C = len(class_names)
    cm = np.zeros((C, C), dtype=int)
    cm_present = confusion_matrix(y_true, y_pred, labels=labels)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            cm[li, lj] = cm_present[i, j]

    return {
        "loss": loss,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_matrix": cm,
    }