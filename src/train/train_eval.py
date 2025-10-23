# -*- coding: utf-8 -*-
import torch
from torch import nn
from tqdm import tqdm

@torch.no_grad()
def _batch_f1_from_logits(logits: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> float:
    """
    简单 F1 计算：
    - CrossEntropy: 取 argmax 作为预测类别
    - BCE:          sigmoid>=0.5 为正类
    """
    if isinstance(criterion, nn.BCEWithLogitsLoss):
        probs = torch.sigmoid(logits).view(-1)
        preds = (probs >= 0.5).long()
        target = y.view(-1).long()
    else:
        preds = logits.argmax(dim=1).view(-1)
        target = y.view(-1).long()

    tp = ((preds == 1) & (target == 1)).sum().item()
    fp = ((preds == 1) & (target == 0)).sum().item()
    fn = ((preds == 0) & (target == 1)).sum().item()

    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def train_epoch(model, loader, criterion, optimizer, device):
    """
    期望 DataLoader 的 collate_fn 返回：
    - inputs: float32 tensor [B, ...]
    - labels: long tensor    [B]（适配 CrossEntropy）
    """
    model.train()
    running_loss, running_f1, n_batches = 0.0, 0.0, 0

    for inputs, labels in tqdm(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)

        # CrossEntropyLoss 期望 logits [B, C] & labels [B] (long)
        # 若你用 BCEWithLogitsLoss，请保证 logits [B,1] 且 labels [B,1] float
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_f1   += _batch_f1_from_logits(logits.detach(), labels.detach(), criterion)
        n_batches    += 1

    return running_loss / max(n_batches, 1), running_f1 / max(n_batches, 1)

@torch.no_grad()
def eval_model(model, loader, criterion, device, cm=False):
    model.eval()
    running_loss, running_f1, n_batches = 0.0, 0.0, 0

    for inputs, labels in tqdm(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs)
        loss = criterion(logits, labels)

        running_loss += loss.item()
        running_f1   += _batch_f1_from_logits(logits, labels, criterion)
        n_batches    += 1

    return running_loss / max(n_batches, 1), running_f1 / max(n_batches, 1)
