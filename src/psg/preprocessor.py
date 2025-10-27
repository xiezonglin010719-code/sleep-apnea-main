# -*- coding: utf-8 -*-
"""
PSG 预处理（时序版：输出 (N_segments, T=pad_to_frames, F=6*#modalities)）
- 读取同一患者多段 EDF [001..005]，整夜拼接
- 健壮的通道选择（含 Effort THO 兜底）
- 事件敏感帧特征：RMS、ΔRMS、ZCR、峰值频率、频谱带宽、振幅范围（每模态6维）
- 标签对齐按覆盖比例（默认阈值 ratio=0.3），可选仅呼吸事件
"""

import os
import re
import warnings
from typing import Tuple, Dict, List
import numpy as np

from scipy.signal import butter, lfilter, resample
from scipy.ndimage import uniform_filter1d
from scipy.stats import zscore
from tqdm import tqdm

from src.preprocessing.steps.configpsg import Config, PSGConfig
from src.psg.rml_parser import RMLParser
from src.psg.rml_align_utils import build_label_mapping

warnings.filterwarnings(
    "ignore",
    message=r"Loading an EDF with mixed sampling frequencies and preload=False.*",
    category=RuntimeWarning,
)

try:
    import pyedflib
    _HAS_PYEDFLIB = True
except Exception:
    _HAS_PYEDFLIB = False


# ---------- 工具 ----------
def _zcr(frames: np.ndarray) -> np.ndarray:
    """零交叉率：每帧符号变化次数 / 帧长"""
    s = np.sign(frames)
    s[s == 0] = 1
    return (np.diff(s, axis=1) != 0).sum(axis=1) / frames.shape[1]

def _rms(frames: np.ndarray) -> np.ndarray:
    """帧内 RMS"""
    return np.sqrt(np.mean(frames ** 2, axis=1) + 1e-8)

def _peak_freq_and_bw(frames: np.ndarray, sr: int):
    """峰值频率与近似带宽"""
    fft = np.fft.rfft(frames, axis=1)
    mag = np.abs(fft)
    psd = mag ** 2
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sr)
    total = np.sum(psd, axis=1, keepdims=True) + 1e-8
    peak_idx = np.argmax(psd, axis=1)
    peak_freq = freqs[peak_idx]
    centroid = (np.sum(psd * freqs[None, :], axis=1, keepdims=True) / total).squeeze(1)
    bw = np.sqrt(np.sum(((freqs[None, :] - centroid[:, None]) ** 2) * psd, axis=1) / total.squeeze(1))
    return peak_freq.astype(np.float32), bw.astype(np.float32)

def _amp_range(frames: np.ndarray) -> np.ndarray:
    """振幅摆动：max - min"""
    return (np.max(frames, axis=1) - np.min(frames, axis=1)).astype(np.float32)


class PSGEvent:
    def __init__(self, index: int, patient_id: str, start: float, end: float, label: int, features: np.ndarray):
        self.index = index
        self.patient_id = patient_id
        self.start = start
        self.end = end
        self.label = label
        self.features = features  # (T, F)


class PSGPreprocessor:
    # 参与时序建模的模态（可通过 YAML target_channels 精选，否则 fallback 到这里）
    TIMED_MODALITIES = [
        "EEG", "EOG", "EMG", "ECG", "Flow Patient", "Effort ABD", "Effort THO"
    ]
    SKIP_AUDIO_KEYS = ["snore", "snoring", "mic", "microphone", "sound", "tracheal"]

    MODALITY_ALIASES: Dict[str, List[str]] = {
        "EEG": ["eeg"],
        "EOG": ["eog"],
        "EMG": ["emg"],
        "ECG": ["ecg", "ekg"],
        "Flow Patient": ["flow patient", "thermistor", "pressure cannula", "flow", "airflow", "nasal", "oro", "oral"],
        "Effort ABD": ["effort abd", "abd", "abdominal"],
        "Effort THO": ["effort tho", "tho", "thor", "thoracic", "effort thorax"]
    }

    def __init__(self, config: Config):
        if not _HAS_PYEDFLIB:
            raise RuntimeError("需要 pyEDFlib：pip install pyEDFlib")

        self.config: Config = config
        self.psg_cfg: PSGConfig = config.psg
        self.raw_psg_dir = config.paths.raw_psg_dir
        self.raw_rml_dir = self.raw_psg_dir.replace("edf", "rml")
        self.processed_dir = config.paths.processed_psg_dir
        os.makedirs(self.processed_dir, exist_ok=True)

        # 标签映射 & 解析器
        self.label_mapping = build_label_mapping(self.psg_cfg.event_labels)
        if "Normal" not in self.label_mapping:
            self.label_mapping["Normal"] = 0
        self.rml_parser = RMLParser(event_mapping=self.label_mapping)

        # 帧化参数
        self.frame_len_s: float = float(self.psg_cfg.frame_length_s)
        self.frame_hop_s: float = float(self.psg_cfg.frame_hop_s)
        self.pad_to_frames: int = int(self.psg_cfg.pad_to_frames)
        # 注意：per_frame_features 固定为 6（下方设计），若你要扩增，请同步调整模型 input_dim
        self.per_frame_features: int = 6

        # 兜底：若配置没写 Effort THO 的采样率/滤波，沿用 ABD
        rates = self.psg_cfg.channel_sampling_rates
        cuts = self.psg_cfg.bandpass_cutoffs
        if "Effort THO" not in rates:
            rates["Effort THO"] = rates.get("Effort ABD", 100)
        if "Effort THO" not in cuts:
            cuts["Effort THO"] = cuts.get("Effort ABD", (0.1, 2.0))

        wanted = [m for m in getattr(self.psg_cfg, "target_channels", []) if m in self.TIMED_MODALITIES] \
                 or self.TIMED_MODALITIES[:]
        print(f"[PSGPreprocessor] 目标通道: {wanted}")

    # ------- 基础处理 -------
    @staticmethod
    def _butter_bandpass_filter(data: np.ndarray, low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
        nyq = 0.5 * fs
        lo = max(1e-6, low / nyq)
        hi = min(0.999, high / nyq)
        b, a = butter(order, [lo, hi], btype='band')
        return lfilter(b, a, data)

    def _preprocess_to_target_sr(self, signal: np.ndarray, modality: str, orig_sr: float) -> Tuple[np.ndarray, int]:
        target_sr = int(self.psg_cfg.channel_sampling_rates.get(modality, self.psg_cfg.channel_sampling_rates.get("Effort ABD", 100)))
        x = signal.astype(np.float32, copy=False)

        # 重采样
        if int(orig_sr) != target_sr and len(x) > 0:
            n_samples = max(1, int(round(len(x) * target_sr / float(orig_sr))))
            x = resample(x, n_samples).astype(np.float32, copy=False)

        # 滤波
        low, high = self.psg_cfg.bandpass_cutoffs.get(modality, (0.1, 2.0))
        if len(x) > target_sr:  # 至少 1s 再滤波
            order = 6 if modality in ("Effort ABD", "Effort THO") else 4
            x = self._butter_bandpass_filter(x, low, high, target_sr, order=order)

        # 标准化
        if modality in ("Effort ABD", "Effort THO"):
            window = int(target_sr * 5)
            if len(x) >= window:
                mean = uniform_filter1d(x, size=window, mode='nearest')
                sq_mean = uniform_filter1d(x ** 2, size=window, mode='nearest')
                std = np.sqrt(np.maximum(sq_mean - mean ** 2, 1e-8))
                x = (x - mean) / (std + 1e-8)
            elif len(x) > 1:
                x = zscore(x, ddof=1)
        else:
            if len(x) > 1:
                x = zscore(x, ddof=1)

        np.nan_to_num(x, copy=False)
        return x.astype(np.float32, copy=False), target_sr

    @staticmethod
    def _frame_signal(signal: np.ndarray, sr: int, frame_len_s: float, frame_hop_s: float) -> np.ndarray:
        L = len(signal)
        fl = int(round(frame_len_s * sr))
        fh = int(round(frame_hop_s * sr))
        if L < fl:  # 不足一帧
            return np.zeros((0, fl), dtype=np.float32)
        n_frames = (L - fl) // fh + 1
        if n_frames <= 0:
            return np.zeros((0, fl), dtype=np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(signal, window_shape=fl)[::fh]
        return frames.astype(np.float32, copy=False)

    def _per_frame_feats(self, frames: np.ndarray, sr: int, modality: str) -> np.ndarray:
        """
        每帧 6 维：RMS, ΔRMS, ZCR, 峰值频率, 近似带宽, 振幅范围
        这些对低频呼吸波形的能量变化、节律变化、阻塞时幅度/频谱变化更敏感。
        """
        T = frames.shape[0]
        if T == 0:
            return np.zeros((0, self.per_frame_features), dtype=np.float32)

        feat = np.zeros((T, self.per_frame_features), dtype=np.float32)
        rms = _rms(frames)
        drms = np.concatenate([[0.0], np.diff(rms)]).astype(np.float32)
        zcr = _zcr(frames)
        peak_f, bw = _peak_freq_and_bw(frames, sr)
        ar = _amp_range(frames)

        feat[:, 0] = rms
        feat[:, 1] = drms
        feat[:, 2] = zcr
        feat[:, 3] = peak_f
        feat[:, 4] = bw
        feat[:, 5] = ar
        return feat

    @staticmethod
    def _resize_time_axis(x_tf: np.ndarray, target_T: int) -> np.ndarray:
        T, F = x_tf.shape
        if T == target_T:
            return x_tf
        if T < target_T:
            out = np.zeros((target_T, F), dtype=x_tf.dtype)
            out[:T] = x_tf
            return out
        # 线性插值压缩到 target_T
        x_old = np.linspace(0.0, 1.0, num=T, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=target_T, dtype=np.float32)
        out = np.zeros((target_T, F), dtype=np.float32)
        for f in range(F):
            out[:, f] = np.interp(x_new, x_old, x_tf[:, f]).astype(np.float32)
        return out

    # 放回 PSGPreprocessor 类中（例如放在 __init__ 之后、process_patient 之前/之后都可）

    @staticmethod
    def _frame_signal_1d(signal: np.ndarray, sr: int, frame_len_s: float, frame_hop_s: float) -> np.ndarray:
        """
        将 1D 信号按 (frame_len_s, frame_hop_s) 帧化为 (T, frame_len_samples)。
        - 长度不足一帧返回 shape (0, frame_len) 的空数组
        """
        frame_len = int(round(frame_len_s * sr))
        frame_hop = int(round(frame_hop_s * sr))
        if frame_len <= 0 or frame_hop <= 0 or len(signal) < frame_len:
            return np.zeros((0, max(frame_len, 1)), dtype=np.float32)

        # 用滑动窗口 + 步长（stride trick），效率高
        frames = np.lib.stride_tricks.sliding_window_view(signal, window_shape=frame_len)[::frame_hop]
        # 可能因为整除问题导致最后不足一帧的部分未被覆盖，这是预期的
        return frames.astype(np.float32, copy=False)

    def _per_frame_feats(self, frames: np.ndarray, sr: int, modality: str) -> np.ndarray:
        """
        从每帧提 6 维特征，与你之前代码保持一致：
          - 4 个频带功率占比 + 频谱质心 + “带宽”/峰频率与峰幅（模态不同略有差异）
        返回 shape = (T, self.per_frame_features)
        """
        T = frames.shape[0]
        if T == 0:
            return np.zeros((0, self.per_frame_features), dtype=np.float32)

        # 频域
        fft = np.fft.rfft(frames, axis=1)
        mag = np.abs(fft)
        psd = mag ** 2
        freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sr)
        total = np.sum(psd, axis=1, keepdims=True) + 1e-8

        feats = np.zeros((T, self.per_frame_features), dtype=np.float32)

        # 不同模态的带划分/额外两维
        if modality in ("EEG", "EOG", "EMG", "ECG"):
            # 4 个 EEG 常用带
            bands = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
            centroid = (np.sum(psd * freqs[None, :], axis=1, keepdims=True) / total).squeeze(1)
            bw = np.sqrt(np.sum(((freqs[None, :] - centroid[:, None]) ** 2) * psd, axis=1) / total.squeeze(1))
            feats[:, 4] = centroid
            feats[:, 5] = bw
        else:
            # 呼吸相关：低频带
            bands = [(0.1, 0.4), (0.4, 0.7), (0.7, 1.0), (1.0, 2.0)]
            peak_freqs = freqs[np.argmax(psd, axis=1)]
            feats[:, 4] = peak_freqs
            feats[:, 5] = np.max(np.abs(frames), axis=1)

        # 4 个带的功率占比
        for i, (lo, hi) in enumerate(bands[:4]):
            mask = (freqs >= lo) & (freqs < hi)
            if not np.any(mask):
                continue
            feats[:, i] = (np.sum(psd[:, mask], axis=1) / total.squeeze(1)).astype(np.float32)

        return feats.astype(np.float32, copy=False)

    @staticmethod
    def _resize_time_axis(x_tf: np.ndarray, target_T: int) -> np.ndarray:
        """
        将 (T,F) 时间轴调整为 target_T：
          - T < target_T: 右侧 0 填充
          - T > target_T: 按时间线性插值
        """
        T, F = x_tf.shape
        if T == target_T:
            return x_tf
        if T < target_T:
            out = np.zeros((target_T, F), dtype=x_tf.dtype)
            out[:T] = x_tf
            return out

        # T > target_T → 线性插值到 target_T
        x_old = np.linspace(0.0, 1.0, num=T, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=target_T, dtype=np.float32)
        out = np.zeros((target_T, F), dtype=np.float32)
        for f in range(F):
            out[:, f] = np.interp(x_new, x_old, x_tf[:, f]).astype(np.float32)
        return out

    # ------- 主流程 -------
    def process_patient(self, patient_id: str):
        """
        读取同一患者的多个 EDF（[001]~[005]等）→ 预处理到目标采样率 → 逐文件分段并帧化 → 拼接整夜 (N,T,F)
        解析 RML（整夜绝对时间，单位：秒）→ 按整夜 total_duration 对齐 → 生成整夜段标签 → 保存
        返回:
            features: np.ndarray, 形状 (N, T, F)
            labels:   np.ndarray, 形状 (N,)
        """
        import re
        import numpy as np
        import os

        # ---------- 1) 定位 EDF/RML ----------
        psg_dir = os.path.join(self.raw_psg_dir, patient_id)
        if not os.path.isdir(psg_dir):
            raise FileNotFoundError(f"PSG子目录不存在: {psg_dir}")

        # 取出该患者所有 edf，按文件名中的 [001]…[005] 排序
        edf_files = [f for f in os.listdir(psg_dir) if f.endswith(".edf") and patient_id in f]
        if not edf_files:
            raise FileNotFoundError(f"未找到EDF文件: {psg_dir}")

        def _edf_index(fname: str) -> int:
            m = re.search(r"\[(\d{3})\]", fname)
            return int(m.group(1)) if m else 1

        edf_files.sort(key=_edf_index)

        rml_dir = os.path.join(self.raw_rml_dir, patient_id)
        if not os.path.isdir(rml_dir):
            raise FileNotFoundError(f"RML子目录不存在: {rml_dir}")
        rml_files = [f for f in os.listdir(rml_dir) if f.endswith(".rml") and patient_id in f]
        if not rml_files:
            raise FileNotFoundError(f"未找到RML文件: {rml_dir}")
        rml_files.sort()
        rml_path = os.path.join(rml_dir, rml_files[0])

        # ---------- 2) 读取 RML（整夜全局事件） ----------
        print(f"加载RML标签文件: {rml_path}")
        raw_events = self.rml_parser.parse_rml(rml_path)  # 期望: [{'start':sec,'end':sec,'type'或'label':...}, ...]
        # 统一事件字典格式，确保 'type' 是配置里能识别的键（允许规范化匹配）
        rev_map = {v: k for k, v in self.label_mapping.items()}

        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", str(s).lower())

        events_dicts = []
        for e in raw_events:
            # 取“可读类型名”
            if isinstance(e, dict) and isinstance(e.get("type"), str) and e["type"]:
                lbl_str = e["type"]
            elif isinstance(e, dict) and isinstance(e.get("label"), int):
                lbl_str = rev_map.get(e["label"], "Normal")
            else:
                lbl_str = "Normal"

            # 尝试把 lbl_str 规范化后与 label_mapping 的键对齐
            n = _norm(lbl_str)
            mapped = lbl_str
            if lbl_str not in self.label_mapping:
                for k in self.label_mapping.keys():
                    if _norm(k) == n:
                        mapped = k
                        break

            try:
                st = float(e["start"])
                et = float(e["end"])
            except Exception:
                continue
            if et <= st:
                continue

            events_dicts.append({"start": st, "end": et, "type": mapped})

        print(f"[RML] 全夜事件数: {len(events_dicts)}")
        # 调试分布
        from collections import Counter
        print("[RML] 事件类型分布:", Counter([_norm(x["type"]) for x in events_dicts]) or "{}")

        # ---------- 3) 遍历 EDF：通道选择 → 预处理 → 30s 分段 → 每段帧化并拼特征 ----------
        seg_len_s = float(self.psg_cfg.segment_length)  # 例：30
        step_s = seg_len_s * (1.0 - float(self.psg_cfg.segment_overlap))  # 例：15
        print(
            f"分段配置: 每段{seg_len_s}s，步长{step_s}s；帧长{self.frame_len_s}s，步长{self.frame_hop_s}s，目标帧数{self.pad_to_frames}")

        processed_events = []
        total_duration_all = 0.0
        offset_sec = 0.0

        for fname in edf_files:
            edf_path = os.path.join(psg_dir, fname)
            if not _HAS_PYEDFLIB:
                raise RuntimeError("需要安装 pyEDFlib：pip install pyEDFlib")
            f = pyedflib.EdfReader(edf_path)
            try:
                n_sig = f.signals_in_file
                labels = [f.getLabel(i) for i in range(n_sig)]
                labels_lc = [lab.lower() for lab in labels]
                srs = [float(f.getSampleFrequency(i)) for i in range(n_sig)]
                duration_sec = float(f.getFileDuration())
                print(f"加载PSG文件(pyEDFlib): {edf_path} (时长: {duration_sec:.1f}s, 通道: {n_sig})")

                # ---- 通道选择（跳过声学） ----
                def pick_index_for_modality(mod: str) -> int:
                    keys = self.MODALITY_ALIASES.get(mod, [mod.lower()])
                    cands = []
                    for i, lab in enumerate(labels_lc):
                        if any(sk in lab for sk in self.SKIP_AUDIO_KEYS):
                            continue
                        if any(k in lab for k in keys):
                            cands.append(i)
                    return cands[0] if cands else -1

                wanted = [m for m in getattr(self.psg_cfg, "target_channels", []) if m in self.TIMED_MODALITIES] \
                         or self.TIMED_MODALITIES[:]
                pick_map = {}
                for modality in wanted:
                    idx = pick_index_for_modality(modality)
                    if idx >= 0:
                        pick_map[modality] = idx
                    else:
                        print(f"警告: 未找到通道 {modality}，将跳过此模态。")
                if not pick_map:
                    raise ValueError(f"文件{fname}无可用通道（在 {self.TIMED_MODALITIES} 内未匹配到）")

                print("通道映射：")
                for mod, i in pick_map.items():
                    print(f"  {mod} -> {labels[i]} (Fs={srs[i]} Hz)")

                # ---- 读取并预处理各通道 ----
                channel_data = {}
                for mod, ch_idx in pick_map.items():
                    if any(sk in labels_lc[ch_idx] for sk in self.SKIP_AUDIO_KEYS):
                        continue
                    sig = f.readSignal(ch_idx).astype(np.float32, copy=False)
                    # 防止极端溢出
                    np.clip(sig, -1e6, 1e6, out=sig)
                    x, target_sr = self._preprocess_to_target_sr(sig, modality=mod, orig_sr=srs[ch_idx])
                    channel_data[mod] = (x, target_sr)

                if not channel_data:
                    raise ValueError(f"文件{fname}：预处理后无可用通道")

                # ---- 逐段生成特征（绝对时间轴） ----
                n_steps = int(np.floor((duration_sec - seg_len_s) / max(step_s, 1e-6))) + 1
                for k in range(n_steps):
                    t0 = k * step_s
                    t1 = t0 + seg_len_s
                    if t1 > duration_sec:
                        break

                    per_mod_TF = []
                    for mod in self.TIMED_MODALITIES:
                        if mod not in channel_data:
                            per_mod_TF.append(
                                np.zeros((self.pad_to_frames, self.per_frame_features), dtype=np.float32)
                            )
                            continue

                        x, sr = channel_data[mod]
                        start = int(round(t0 * sr))
                        stop = start + int(round(seg_len_s * sr))
                        if stop > len(x):
                            seg = np.zeros((int(round(seg_len_s * sr)),), dtype=np.float32)
                            end = max(0, len(x) - start)
                            if end > 0:
                                seg[:end] = x[start:start + end]
                        else:
                            seg = x[start:stop]

                        frames = self._frame_signal_1d(seg, sr, self.frame_len_s, self.frame_hop_s)
                        if frames.size == 0:
                            feats_tf = np.zeros((self.pad_to_frames, self.per_frame_features), dtype=np.float32)
                        else:
                            feats_tf = self._per_frame_feats(frames, sr=sr, modality=mod)
                            feats_tf = self._resize_time_axis(feats_tf, self.pad_to_frames)
                        per_mod_TF.append(feats_tf)

                    seg_TF = np.concatenate(per_mod_TF, axis=1)  # (T=pad_to_frames, F=模态数*per_frame_features)

                    abs_t0 = offset_sec + t0
                    abs_t1 = offset_sec + t1
                    processed_events.append(
                        PSGEvent(
                            index=len(processed_events),
                            patient_id=patient_id,
                            start=abs_t0,  # 绝对时间
                            end=abs_t1,  # 绝对时间
                            label=-1,
                            features=seg_TF
                        )
                    )

                # 累加偏移与总时长
                offset_sec += duration_sec
                total_duration_all += duration_sec

            finally:
                try:
                    f._close()
                except Exception:
                    pass
                del f

        if not processed_events:
            raise ValueError(f"患者{patient_id}：没有产生任何分段特征")

        # ---------- 4) 整夜对齐（按覆盖比例，允许规范化名匹配） ----------
        def _align_by_ratio(events, seg_len: float, ratio_th: float = 0.10):
            """
            一个段与某类事件的时间重叠占段长的比例 >= ratio_th 就赋该事件标签，否则 0(Normal)。
            事件名通过规范化映射到 config.psg.event_labels 的键（避免空格/大小写差异）。
            """
            N = len(processed_events)
            labels = np.zeros((N,), dtype=np.int64)

            # 构建“规范化名 -> 原始键”的映射（排除 Normal）
            canon_map = {_norm(k): k for k in self.label_mapping.keys() if _norm(k) != _norm("Normal")}

            for i, seg in enumerate(processed_events):
                s0, s1 = seg.start, seg.end
                best_label, best_ratio = 0, 0.0
                for ev in events:
                    et0, et1 = float(ev["start"]), float(ev["end"])
                    ntyp = _norm(ev["type"])
                    if ntyp not in canon_map:
                        continue  # 配置未定义的事件，忽略
                    key = canon_map[ntyp]
                    lbl = self.label_mapping.get(key, 0)

                    ov = max(0.0, min(s1, et1) - max(s0, et0))
                    ratio = ov / float(seg_len)
                    if ratio > best_ratio:
                        best_ratio, best_label = ratio, lbl
                labels[i] = best_label if best_ratio >= ratio_th else 0
            return labels

        segment_labels = _align_by_ratio(
            events=events_dicts,
            seg_len=seg_len_s,
            ratio_th=0.10,  # 先用 0.10；如仍偏少，可进一步下调至 0.05 试验
        )

        # ---------- 5) 打包保存 ----------
        features = np.stack([e.features for e in processed_events], axis=0)  # (N, T, F)
        N = features.shape[0]
        labels_arr = np.asarray(segment_labels, dtype=np.int64)
        if len(labels_arr) > N:
            labels_arr = labels_arr[:N]
        elif len(labels_arr) < N:
            pad = np.zeros((N - len(labels_arr),), dtype=np.int64)
            labels_arr = np.concatenate([labels_arr, pad], axis=0)

        # 统计分布
        uniq, cnt = np.unique(labels_arr, return_counts=True)
        print("标签分布:", dict(zip(uniq.tolist(), cnt.tolist())))

        start_times = np.array([e.start for e in processed_events], dtype=np.float32)
        end_times = np.array([e.end for e in processed_events], dtype=np.float32)

        os.makedirs(self.processed_dir, exist_ok=True)
        save_path = os.path.join(self.processed_dir, f"{patient_id}_features.npz")
        np.savez_compressed(
            save_path,
            features=features,  # (N, T, F)
            labels=labels_arr,  # (N,)
            start_times=start_times,
            end_times=end_times
        )
        print(f"患者{patient_id}处理完成: 特征{features.shape} (N,T,F)，标签{len(labels_arr)} → {save_path}")

        return features, labels_arr

    def batch_process(self, ids: List[str]) -> None:
        for pid in tqdm(ids, desc="PSG数据预处理"):
            try:
                self.process_patient(pid)
            except Exception as e:
                print(f"处理患者{pid}失败: {e}")
