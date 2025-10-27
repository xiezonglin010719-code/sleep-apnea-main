# src/federated/server.py
import os
import copy
from typing import Dict, List, Tuple, OrderedDict as OrderedDictType

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

from src.federated.client import FederatedClient
from src.utils.metrics import plot_confusion_matrix, evaluate_classification


# ===== 简单的 npz 数据集封装 =====
class NPZDataset(Dataset):
    """
    期望 npz 里包含：
      - features: (N, ...)  # 可是 (N, 128, 36) 或 (N, D)
      - labels:   (N,)      # 可选，没有就用 -1
    """
    def __init__(self, npz_path: str):
        arr = np.load(npz_path)
        self.x = arr["features"]
        self.y = arr["labels"] if "labels" in arr.files else np.full((self.x.shape[0],), -1, dtype=np.int64)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        # 统一为 Long 索引
        yy = self.y[idx]
        y = torch.tensor(yy, dtype=torch.long) if np.ndim(yy) == 0 else torch.tensor(yy).long().view(-1)[0]
        return x, y


def _make_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """基于标签频次构造样本级权重 = 1/freq[label] 的均衡采样器"""
    labels = labels.astype(int).reshape(-1)
    classes, counts = np.unique(labels, return_counts=True)
    freq = {int(c): int(n) for c, n in zip(classes.tolist(), counts.tolist())}
    sample_w = [1.0 / max(freq.get(int(t), 1), 1) for t in labels.tolist()]
    weights = torch.tensor(sample_w, dtype=torch.float32)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _build_client_loaders(
    client_dir: str,
    batch_size: int,
    num_workers: int = 0,
    balanced_sampler: bool = True
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    从 client_dir 加载 train/val 的 .npz，返回 dataloader 和样本数（用于 FedAvg 加权）
    目录结构示例：
      client_dir/
        train.npz
        val.npz
    """
    train_npz = os.path.join(client_dir, "train.npz")
    val_npz   = os.path.join(client_dir, "val.npz")

    if not os.path.exists(train_npz):
        raise FileNotFoundError(f"未找到 {train_npz}")
    if not os.path.exists(val_npz):
        # 没有验证集时，临时用训练集替代
        val_npz = train_npz

    train_ds = NPZDataset(train_npz)
    val_ds   = NPZDataset(val_npz)

    # 训练集：可选均衡采样
    if balanced_sampler:
        sampler = _make_balanced_sampler(train_ds.y)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, drop_last=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=False)

    # 验证集：顺序加载
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)

    return train_loader, val_loader, len(train_ds), len(val_ds)


class FederatedServer:
    def __init__(self, global_model: torch.nn.Module, config, device: torch.device):
        fed = getattr(config, "federated", None)
        self.batch_size: int = int(getattr(fed, "batch_size", 32))
        self.config = config
        self.best_f1 = -1.0
        self.best_model_path = os.path.join(config.paths.global_model_dir, "best_model.pt")
        os.makedirs(config.paths.global_model_dir, exist_ok=True)

        # 类名解析：优先顶层 property，否则从 psg 里拿，最后兜底
        self.class_names = (
            getattr(config, "class_names", None)
            or getattr(getattr(config, "psg", object()), "class_names", None)
            or ["Class 0", "Class 1", "Class 2"]
        )

        self.global_rounds: int = int(getattr(fed, "global_rounds", 15))
        self.client_fraction: float = float(getattr(fed, "client_fraction", 1.0))
        self.num_clients: int = int(getattr(fed, "num_clients", 5))
        self.use_balanced_sampler: bool = bool(getattr(fed, "balanced_sampler", True))

        self.global_model = global_model.to(device)
        self.device = device
        self.clients_root: str = config.paths.federated_data_dir
        self.path = os.path.join(os.path.dirname(config.paths.global_model_dir), "evaluation_results")

        # ✅ classmethod 直接用 self 调用即可（不会再触发 staticmethod 套娃问题）
        self.class_weights = self._compute_global_class_weights(
            clients_root=self.clients_root,
            num_clients=self.num_clients,
            num_classes=len(self.class_names),
            device=self.device,
        )
        if self.class_weights is not None:
            print(f"[INFO] class_weights = {self.class_weights.detach().cpu().numpy().tolist()}")

    # --- 在 FederatedServer 类里新增 ---
    @torch.no_grad()
    def _warmup_initialize_global_model(self, clients_root: str, batch_size: int = 2):
        """
        用任意一个客户端的数据，跑一次前向，触发 Lazy 模块参数初始化。
        """
        # 找到第一个有数据的客户端
        for cid in range(self.num_clients):
            p = os.path.join(clients_root, str(cid), "train.npz")
            if not os.path.isfile(p):
                continue
            arr = np.load(p)
            X = arr["features"]
            if X.ndim < 2:
                continue
            # 取一个小 batch，转成 (B,T,F)
            xb = torch.tensor(X[:batch_size], dtype=torch.float32, device=self.device)
            # 如果你的张量是 (N,F) 或 (N,T,F) 以外的形状，这里按你的模型期望调整
            if xb.ndim == 2:
                xb = xb.unsqueeze(1)  # (B,1,F) -> 具体看你的 forward 定义，通常是 (B,T,F)
            self.global_model.eval()
            _ = self.global_model(xb)  # 跑一次前向，实化 Lazy 参数
            return  # 成功即返回
        # 如果走到这，说明没有可用客户端样本（通常不会发生）
        print("[WARN] _warmup_initialize_global_model: 未找到可用的客户端样本，跳过 Lazy 初始化。")

    @classmethod
    def _compute_global_class_weights(
        cls, clients_root: str, num_clients: int, num_classes: int, device: torch.device
    ) -> torch.Tensor:
        counts = np.zeros(num_classes, dtype=np.int64)
        for cid in range(num_clients):
            p = os.path.join(clients_root, str(cid), "train.npz")
            if not os.path.isfile(p):
                continue
            y = np.load(p)["labels"].astype(int).reshape(-1)
            cls_ids, cnts = np.unique(y, return_counts=True)
            for c, n in zip(cls_ids.tolist(), cnts.tolist()):
                if 0 <= c < num_classes:
                    counts[c] += int(n)

        if counts.sum() == 0:
            return None

        # 逆频 + 缓和指数，避免过度放大少数类
        counts = np.maximum(counts, 1)
        weights = (counts.sum() / counts) ** 0.7  # 让少数类 > 多数类
        return torch.tensor(weights, dtype=torch.float32, device=device)

    @staticmethod
    def _list_existing_client_ids(root_dir: str, max_clients: int) -> List[int]:
        ids = []
        for i in range(max_clients):
            d = os.path.join(root_dir, str(i))
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "train.npz")):
                ids.append(i)
        return ids

    @staticmethod
    def _build_global_val_loader(root_dir: str, client_ids: List[int], batch_size: int) -> DataLoader:
        """把所有客户端的 val.npz（若无则用 train.npz）合并成一个全局验证集"""
        val_sets: List[Dataset] = []
        for cid in client_ids:
            cdir = os.path.join(root_dir, str(cid))
            vpath = os.path.join(cdir, "val.npz")
            if not os.path.isfile(vpath):
                vpath = os.path.join(cdir, "train.npz")
            if os.path.isfile(vpath):
                val_sets.append(NPZDataset(vpath))
        if not val_sets:
            raise FileNotFoundError("未找到任意客户端的 {val,train}.npz 用于构建全局验证集")
        concat = ConcatDataset(val_sets)
        return DataLoader(concat, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    @staticmethod
    def _weighted_fedavg(states: List[OrderedDictType[str, torch.Tensor]], weights: List[float]) -> OrderedDictType[str, torch.Tensor]:
        """
        只对浮点参数做加权平均；非浮点（如 BN 的 num_batches_tracked: Long）直接拷贝第一个客户端的值。
        weights 应已归一化（和为 1）。
        """
        assert len(states) > 0, "没有可聚合的客户端权重"
        from collections import OrderedDict
        agg = OrderedDict()

        # 初始化
        for k, v0 in states[0].items():
            if torch.is_floating_point(v0):
                agg[k] = torch.zeros_like(v0, dtype=torch.float32)
            else:
                agg[k] = v0.clone()

        # 加权和（仅浮点）
        for st, w in zip(states, weights):
            for k, t in st.items():
                if torch.is_floating_point(t):
                    agg[k] += t.to(torch.float32) * float(w)
        return agg

    def train(self):
        os.makedirs(self.path, exist_ok=True)

        all_client_ids = self._list_existing_client_ids(self.clients_root, self.num_clients)
        if not all_client_ids:
            raise FileNotFoundError(f"在 {self.clients_root} 下未发现任何客户端数据目录")
        print(f"[INFO] 可用客户端: {all_client_ids}")

        global_val_loader = self._build_global_val_loader(self.clients_root, all_client_ids, self.batch_size)
        self._warmup_initialize_global_model(self.clients_root, batch_size=2)

        rng = np.random.default_rng(seed=getattr(self.config, "random_seed", 42))

        for rnd in range(1, self.global_rounds + 1):
            print(f"\n===== 全局轮次 {rnd}/{self.global_rounds} =====")
            num_sel = max(1, int(np.ceil(len(all_client_ids) * float(self.client_fraction))))
            selected_ids = rng.choice(all_client_ids, size=num_sel, replace=False)
            print(f"[ROUND {rnd}] 选中的客户端: {sorted(selected_ids.tolist())}")

            client_states: List[OrderedDictType[str, torch.Tensor]] = []
            client_sizes: List[int] = []

            for cid in selected_ids:
                cdir = os.path.join(self.clients_root, str(cid))
                train_loader, val_loader, n_train, _ = _build_client_loaders(
                    client_dir=cdir,
                    batch_size=self.batch_size,
                    num_workers=0,
                    balanced_sampler=self.use_balanced_sampler
                )

                client = FederatedClient(
                    client_id=cid,
                    model=copy.deepcopy(self.global_model),
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=self.config,
                    device=self.device,
                    class_weights=self.class_weights,  # 你已在 FederatedClient 接口里加上了这个参数
                )

                local_state, metrics = client.train(self.global_model.state_dict())
                client_states.append(local_state)
                client_sizes.append(int(n_train))
                print(f"[Client {cid}] 训练完成，样本数={n_train}, 指标={metrics}")

            if client_states:
                sizes = np.array(client_sizes, dtype=np.float64)
                weights = sizes / max(sizes.sum(), 1.0)
                aggregated_state = self._weighted_fedavg(client_states, weights.tolist())

                current = self.global_model.state_dict()
                filtered = {}
                dropped = []
                for k, v in aggregated_state.items():
                    if k in current and v.shape == current[k].shape:
                        filtered[k] = v.to(dtype=current[k].dtype, device=current[k].device)
                    else:
                        dropped.append(k)
                if dropped:
                    print(f"[WARN] 跳过无法加载的参数 {len(dropped)} 个（键或形状不匹配），示例: {dropped[:6]}")

                missing, unexpected = self.global_model.load_state_dict({**current, **filtered}, strict=False)
                if unexpected:
                    print(f"[INFO] load_state_dict unexpected keys: {unexpected}")
                if missing:
                    print(f"[INFO] load_state_dict missing keys: {missing}")

            print("\n[本轮评估]")
            eval_results = evaluate_classification(
                self.global_model,
                global_val_loader,
                self.device,
                self.class_names
            )
            print(f"验证集损失: {eval_results['loss']:.4f}")
            print(f"Macro F1: {eval_results['f1_macro']:.4f}")
            print(f"Micro F1: {eval_results['f1_micro']:.4f}")
            print("\n分类报告:\n", eval_results['classification_report'])

            cm_path = os.path.join(self.path, f"confusion_matrix_round_{rnd}.png")
            plot_confusion_matrix(
                eval_results['y_true'],
                eval_results['y_pred'],
                self.class_names,
                cm_path
            )
            print(f"混淆矩阵已保存至: {cm_path}")

            if eval_results['f1_macro'] > self.best_f1:
                self.best_f1 = eval_results['f1_macro']
                torch.save({
                    'round': rnd,
                    'model_state_dict': self.global_model.state_dict(),
                    'f1_score': self.best_f1,
                    'eval_results': eval_results
                }, self.best_model_path)
                print(f"最佳模型已更新并保存至: {self.best_model_path} (F1: {self.best_f1:.4f})")

        final_model_path = os.path.join(self.config.paths.global_model_dir, "final_model.pt")
        torch.save(self.global_model.state_dict(), final_model_path)
        print(f"最终模型已保存至: {final_model_path}")

    @torch.no_grad()
    def get_global_model(self) -> torch.nn.Module:
        self.global_model.eval()
        return self.global_model
