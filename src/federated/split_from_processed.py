# src/federated/split_from_processed.py
import os
import glob
import numpy as np
from typing import List, Tuple

from src.preprocessing.steps.configpsg import load_config


def _load_processed_files(processed_dir: str, patient_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 processed_dir 读取若干个 *_features.npz，拼接为 (X_all, y_all)。
    允许 X 是 (N, 128, 36) 或 (N, D)。若没有 labels，自动用 -1 占位。
    """
    X_list, y_list = [], []
    if patient_ids:
        # 只取配置里列出的患者
        paths = [os.path.join(processed_dir, f"{pid}_features.npz") for pid in patient_ids]
    else:
        # 否则扫描所有 *_features.npz
        paths = glob.glob(os.path.join(processed_dir, "*_features.npz"))

    if not paths:
        raise FileNotFoundError(f"在 {processed_dir} 未找到 *_features.npz")

    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"未找到 {p}")
        arr = np.load(p)
        if "features" not in arr.files:
            raise KeyError(f"{p} 缺少 'features' 键")
        X = arr["features"]
        y = arr["labels"] if "labels" in arr.files else np.full((X.shape[0],), -1, dtype=np.int64)
        X_list.append(X)
        y_list.append(y)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0).astype(np.int64)
    print(f"[INFO] 汇总特征: X={X_all.shape}, y={y_all.shape}（labels∈{np.unique(y_all)}）")
    return X_all, y_all


def _train_val_split_idx(n: int, train_ratio: float, val_ratio: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    return train_idx, val_idx


def _partition_iid(y: np.ndarray, num_clients: int, alpha: float = None, seed: int = 42) -> List[np.ndarray]:
    """
    IID 或 Dirichlet 非IID 划分，返回每个 client 的样本索引数组。
    y 可含 -1（未知标签）；非IID 时仅对 ≥0 的类别做 Dirichlet 分配，-1 会均分。
    """
    n = len(y)
    idx_all = np.arange(n)
    rng = np.random.default_rng(seed)

    if alpha is None:
        # 简单均分
        return [s.copy() for s in np.array_split(idx_all, num_clients)]

    # 非IID：对每个已知类别做 Dirichlet 分配
    idx_bins = [[] for _ in range(num_clients)]
    classes = np.array([c for c in np.unique(y) if c >= 0], dtype=int)

    for c in classes:
        c_idx = idx_all[y == c]
        if len(c_idx) == 0:
            continue
        rng.shuffle(c_idx)
        props = rng.dirichlet([alpha] * num_clients)
        counts = np.floor(props * len(c_idx)).astype(int)
        rem = len(c_idx) - counts.sum()
        for i in np.argsort(-props)[:rem]:
            counts[i] += 1
        start = 0
        for i in range(num_clients):
            take = counts[i]
            if take > 0:
                idx_bins[i].append(c_idx[start:start + take])
                start += take

    # 未知标签（-1）均分
    unk = idx_all[y < 0]
    if len(unk) > 0:
        parts = np.array_split(unk, num_clients)
        for i in range(num_clients):
            if len(parts[i]):
                idx_bins[i].append(parts[i])

    return [np.concatenate(b) if len(b) else np.array([], dtype=int) for b in idx_bins]


def split_from_processed(cfg_path: str):
    cfg = load_config(cfg_path)

    processed_dir = cfg.paths.processed_psg_dir
    clients_root  = cfg.paths.federated_data_dir  # e.g. psg_federated/federated/clients
    os.makedirs(clients_root, exist_ok=True)

    ds = cfg.dataset
    fed = cfg.federated

    patient_ids = [str(p) for p in getattr(ds, "patient_ids", [])]
    train_ratio = float(getattr(ds, "train_ratio", 0.7))
    val_ratio   = float(getattr(ds, "val_ratio", 0.15))
    seed        = int(getattr(cfg, "random_seed", 42))

    num_clients = int(getattr(fed, "num_clients", 5))
    split_mode  = str(getattr(ds, "client_data_split", "iid")).lower()
    alpha       = float(getattr(ds, "non_iid_alpha", 0.5)) if split_mode == "non_iid" else None

    # 1) 汇总所有 {pid}_features.npz
    X, y = _load_processed_files(processed_dir, patient_ids)

    # 2) 全局划分 train/val
    train_idx, val_idx = _train_val_split_idx(len(X), train_ratio, val_ratio, seed)
    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx],  y[val_idx]
    print(f"[INFO] 划分: train={Xtr.shape}, val={Xva.shape}")

    # 3) 客户端划分
    tr_parts = _partition_iid(ytr, num_clients, alpha=alpha, seed=seed)
    va_parts = _partition_iid(yva, num_clients, alpha=alpha, seed=seed)

    # 4) 写出每个客户端的数据集
    for cid in range(num_clients):
        cdir = os.path.join(clients_root, str(cid))
        os.makedirs(cdir, exist_ok=True)

        ti, vi = tr_parts[cid], va_parts[cid]
        Xt, yt = (Xtr[ti], ytr[ti]) if len(ti) else (Xtr[:0], ytr[:0])
        Xv, yv = (Xva[vi], yva[vi]) if len(vi) else (Xva[:0], yva[:0])

        np.savez_compressed(os.path.join(cdir, "train.npz"), features=Xt, labels=yt)
        np.savez_compressed(os.path.join(cdir, "val.npz"),   features=Xv, labels=yv)
        print(f"[OK] Client {cid}: train={Xt.shape}, val={Xv.shape} → {cdir}")

    print(f"[DONE] 联邦切分完成，输出目录：{clients_root}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/Users/liyuxiang/Downloads/sleep-apnea-main/psg_federated_config.yaml",
        help="配置文件路径（默认使用本地 psg_federated.yaml）"
    )
    args = parser.parse_args()
    split_from_processed(args.config)
