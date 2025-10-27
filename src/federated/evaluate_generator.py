# -*- coding: utf-8 -*-
"""
联邦声纳生成模型评估脚本（支持无真值占位评估）
- 指标：MSE / Pearson r / 频谱余弦相似度
- 可视化：波形片段对比（自动保存）
- 若验证集无真实声纳（或形状不匹配），自动用随机噪声占位，让评估流程跑通
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from scipy.signal import welch
from typing import Tuple, Any


def _unwrap_model_output(out: Any) -> torch.Tensor:
    """兼容模型 forward 返回 dict 或 Tensor。优先取 dict['sonar']。"""
    if isinstance(out, dict):
        if "sonar" not in out:
            raise ValueError("模型输出为 dict，但缺少 'sonar' 键。")
        return out["sonar"]
    if torch.is_tensor(out):
        return out
    raise TypeError(f"不支持的模型输出类型: {type(out)}")


def _to_numpy_1d(t: torch.Tensor) -> np.ndarray:
    """Tensor -> numpy 1D，保持可比性；如全零或长度太短，避免 pearson 报错"""
    x = t.detach().cpu().float().view(-1).numpy()
    return x


def spectral_similarity(x: np.ndarray, y: np.ndarray, fs: int = 200) -> float:
    """两信号功率谱的余弦相似度（0~1，越大越相似）"""
    # 防止长度过短
    if x.size < 16 or y.size < 16:
        return 0.0
    f1, Pxx = welch(x, fs=fs, nperseg=min(256, x.size))
    f2, Pyy = welch(y, fs=fs, nperseg=min(256, y.size))
    n = min(len(Pxx), len(Pyy))
    if n == 0:
        return 0.0
    Pxx, Pyy = Pxx[:n], Pyy[:n]
    denom = (np.linalg.norm(Pxx) * np.linalg.norm(Pyy) + 1e-8)
    return float(np.dot(Pxx, Pyy) / denom)


def _choose_real_sonar(batch_second: torch.Tensor,
                       gen_sonar: torch.Tensor,
                       allow_fake: bool = True) -> Tuple[torch.Tensor, bool]:
    """
    根据 batch 第二项和生成输出形状来判断是否可作为“真实声纳”。
    若不可用且 allow_fake=True，则返回与 gen_sonar 同形状的随机张量作为占位，并标记 is_fake=True。
    """
    # 如果没有第二项
    if batch_second is None:
        if allow_fake:
            return torch.randn_like(gen_sonar), True
        raise ValueError("验证集中没有真实声纳，也未允许使用占位数据。")

    # 有第二项，但常见情况是“类别标签”（形状 [B] 或 [B,1]）
    # 我们检查它与生成声纳是否同形，若不同则视为不可用 -> 用占位
    if list(batch_second.shape) != list(gen_sonar.shape):
        if allow_fake:
            return torch.randn_like(gen_sonar), True
        raise ValueError(
            f"batch 第二项形状 {tuple(batch_second.shape)} 与生成声纳 {tuple(gen_sonar.shape)} 不匹配，且未允许占位。"
        )

    # 形状匹配即可作为真值
    return batch_second, False


def evaluate_generator(model,
                       dataloader,
                       device,
                       save_dir: str = "evaluation_results",
                       fs: int = 200,
                       allow_fake_target: bool = True,
                       preview_batches: int = 3) -> Tuple[float, float, float]:
    """
    评估声纳生成质量。若无真实声纳，自动用随机张量占位评估（allow_fake_target=True）。
    返回: (avg_mse, avg_pearson_r, avg_spectral_sim)

    dataloader 要求：
      - batch 可以是 (psg_feats,) 或 (psg_feats, something)
      - 若 second 是真实声纳，则形状需与模型输出 "sonar" 一致
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    mse_total = 0.0
    corr_total = 0.0
    spec_total = 0.0
    n_batches = 0
    used_fake_any = False

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 统一拆包：只关心前两项（psg_feats, second）
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    psg_feats, second = batch[0], batch[1]
                else:
                    psg_feats, second = batch[0], None
            else:
                psg_feats, second = batch, None

            psg_feats = psg_feats.to(device).float()
            if second is not None:
                second = second.to(device).float()

            # 前向
            out = model(psg_feats)
            gen_sonar = _unwrap_model_output(out)  # (B, D) 或 (B, T, F) 展平在指标里比较

            # 选取“真实声纳”：若无或不匹配，自动用随机噪声占位
            real_sonar, is_fake = _choose_real_sonar(second, gen_sonar, allow_fake=allow_fake_target)
            used_fake_any = used_fake_any or is_fake

            # 转 numpy 一维
            y_true = _to_numpy_1d(real_sonar)
            y_pred = _to_numpy_1d(gen_sonar)

            # 指标
            mse = mean_squared_error(y_true, y_pred)
            # Pearson 在常数序列时会报 nan，这里做保护
            if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8 or y_true.size < 3:
                corr = 0.0
            else:
                corr = float(pearsonr(y_true, y_pred)[0])
                if not np.isfinite(corr):
                    corr = 0.0
            spec = spectral_similarity(y_true, y_pred, fs=fs)

            mse_total += mse
            corr_total += corr
            spec_total += spec
            n_batches += 1

            # 可视化前 preview_batches 个 batch
            if batch_idx < preview_batches:
                # 取前若干点做可视化，避免图太密
                max_points = min(1000, y_true.size, y_pred.size)
                xs = np.arange(max_points)
                plt.figure(figsize=(8, 4))
                plt.plot(xs, y_true[:max_points], label="Real Sonar" + (" (Fake)" if is_fake else ""), alpha=0.75)
                plt.plot(xs, y_pred[:max_points], label="Generated Sonar", alpha=0.75)
                plt.title(f"Batch {batch_idx} | MSE={mse:.4f}, r={corr:.3f}, spec={spec:.3f}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"waveform_{batch_idx}.png"))
                plt.close()

    if n_batches == 0:
        raise RuntimeError("验证集为空，无法评估。")

    avg_mse = mse_total / n_batches
    avg_corr = corr_total / n_batches
    avg_spec = spec_total / n_batches

    print("\n====================== 评估结果 ======================")
    print(f"平均 MSE:         {avg_mse:.6f}")
    print(f"平均 Pearson r:   {avg_corr:.4f}")
    print(f"平均 频谱相似度:  {avg_spec:.4f}")
    print(f"是否使用了占位真值: {'是' if used_fake_any else '否'}")
    print("=====================================================")

    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"MSE: {avg_mse:.6f}\n")
        f.write(f"Pearson r: {avg_corr:.4f}\n")
        f.write(f"Spectral Sim: {avg_spec:.4f}\n")
        f.write(f"Used Fake Target: {used_fake_any}\n")

    return avg_mse, avg_corr, avg_spec
