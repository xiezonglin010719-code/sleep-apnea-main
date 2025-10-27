# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Optional, Dict, Any
from tqdm import tqdm

class FederatedClient:
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader,
        val_loader,
        config,
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None,
        lr: float = None,
        label_smoothing: float = 0.1,
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        fed = getattr(config, "federated", None)
        self.epochs = int(getattr(fed, "local_epochs", 5))
        self.lr = float(lr if lr is not None else getattr(fed, "learning_rate", 5e-4))

        if class_weights is not None:
            class_weights = class_weights.to(device=device, dtype=torch.float32)
            self.criterion =  FocalLoss(weight=class_weights, gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def _extract_logits(self, out):
        """兼容 dict/tuple/tensor，并在 [B,T,C] 时对 T 维求均值作为分类 logits。"""
        logits = None
        if isinstance(out, dict):
            logits = out.get("logits", None)
            if logits is None:
                for v in out.values():
                    if torch.is_tensor(v):
                        logits = v
                        break
        elif isinstance(out, (list, tuple)):
            for v in out:
                if torch.is_tensor(v):
                    logits = v
                    break
        else:
            logits = out

        if logits is None:
            raise TypeError("Model forward must return logits tensor or dict with 'logits'.")

        if logits.dim() == 3:
            logits = logits.mean(dim=1)
        return logits

    def _one_epoch(self, train: bool = True) -> float:
        loader = self.train_loader if train else self.val_loader
        self.model.train(train)
        total_loss, total_n = 0.0, 0

        pbar = tqdm(loader, desc=f"[Client {self.client_id}] {'train' if train else 'val'}", leave=False)
        for X, y in pbar:
            X = X.to(self.device, non_blocking=True).float()
            y = y.view(-1).to(self.device, non_blocking=True).long()

            out = self.model(X)
            logits = self._extract_logits(out).float()
            loss = self.criterion(logits, y)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_n += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return float(total_loss / max(total_n, 1))

    @torch.no_grad()
    def _evaluate_acc(self) -> float:
        self.model.eval()
        correct, total = 0, 0
        for X, y in self.val_loader:
            X = X.to(self.device).float()
            y = y.view(-1).to(self.device).long()
            logits = self._extract_logits(self.model(X)).float()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        return float(correct / max(total, 1))

    def train(self, global_state_dict: Dict[str, torch.Tensor]):
        missing, unexpected = self.model.load_state_dict(global_state_dict, strict=False)
        if unexpected:
            print(f"[Client {self.client_id}] unexpected keys when loading global: {unexpected}")

        train_loss_last, val_loss_last, val_acc_last = None, None, None
        for _ in range(self.epochs):
            train_loss_last = self._one_epoch(train=True)
            val_loss_last = self._one_epoch(train=False)
            val_acc_last = self._evaluate_acc()

        metrics = {
            "client_id": self.client_id,
            "epochs": self.epochs,
            "batch_size": getattr(self.config.federated, "batch_size", 32),
            "train_loss_last": train_loss_last,
            "val_loss_last": val_loss_last,
            "val_acc_last": val_acc_last,
        }
        return self.model.state_dict(), metrics

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, target):
        # logits: (B,C), target: (B,)
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()