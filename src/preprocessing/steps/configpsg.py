import logging
import os
import torch
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional, Union, Tuple
import yaml

logging.basicConfig(level=logging.INFO)


class EffortABD(BaseModel):
    bandpass_cutoffs: Tuple[float, float] = (0.1, 2.0)
    sampling_rate: int = 100


class EffortTHO(BaseModel):
    bandpass_cutoffs: Tuple[float, float] = (0.1, 2.0)
    sampling_rate: int = 100


class PSGConfig(BaseModel):
    target_channels: List[str] = ["EEG", "EOG", "EMG", "ECG", "Flow Patient", "Effort ABD", "Effort THO"]
    channel_sampling_rates: Dict[str, int] = {
        "EEG": 200, "EOG": 200, "EMG": 200, "ECG": 200, "Flow Patient": 100, "Effort ABD": 100, "Effort THO": 100
    }
    event_labels: Dict[str, int] = {"Normal": 0, "ObstructiveApnea": 1, "Hypopnea": 2}
    segment_length: int = 30
    segment_overlap: float = 0.2
    normalize: bool = True
    bandpass_cutoffs: Dict[str, Tuple[float, float]] = {
        "EEG": (0.5, 30), "EOG": (0.1, 10), "EMG": (10, 100), "ECG": (0.5, 50),
        "Flow Patient": (0.1, 2.0), "Effort ABD": (0.1, 2.0), "Effort THO": (0.1, 2.0)
    }
    # 你的已有字段...
    effort_abd: EffortABD = EffortABD()
    effort_tho: EffortTHO = EffortTHO()
    # 新增分类相关配置
    num_classes: int = 3
    class_names: List[str] = Field(default_factory=lambda: ["normal", "Hypopnea", "ObstructiveApnea"])

    # ☆ 新增：帧化/分段相关字段（与你 YAML 的键完全一致）
    frame_length_s: float = 0.5
    frame_hop_s: float = 0.25
    pad_to_frames: int = 128
    per_frame_features: int = 6

    segment_length: int = 30
    segment_overlap: float = 0.5

    class Config:
        extra = "allow"  # 建议加上，避免今后再加字段时被丢弃

class SonarGeneratorConfig(BaseModel):
    input_dim: int = 256
    output_dim: int = 128
    hidden_layers: List[int] = [512, 256]
    activation: str = "relu"
    dropout_rate: float = 0.3
    generator_type: str = "wgan"


class FederatedConfig(BaseModel):
    global_rounds: int = 15
    local_epochs: int = 5
    num_clients: int = 5
    client_fraction: float = 0.8
    aggregation_strategy: str = "fedavg"
    learning_rate: float = 0.001
    batch_size: int = 32
    differential_privacy: bool = True
    dp_epsilon: float = 8.0
    dp_delta: float = 1e-5
    client_selection: str = "random"


class PathsConfig(BaseModel):
    root_dir: str = "psg_federated"
    raw_psg_dir: str = Field(default=None)
    processed_psg_dir: str = Field(default=None)
    psg_labels_dir: str = Field(default=None)
    federated_data_dir: str = Field(default=None)
    global_model_dir: str = Field(default=None)
    client_models_dir: str = Field(default=None)



    def _init_paths(self):
        base = Path(self.root_dir)
        # 若配置文件未指定路径，则使用默认路径（否则使用配置文件中的路径）
        if not self.raw_psg_dir:
            self.raw_psg_dir = str(base / "raw/psg")
        if not self.processed_psg_dir:
            self.processed_psg_dir = str(base / "processed/psg_features")
        if not self.psg_labels_dir:
            self.psg_labels_dir = str(base / "processed/psg_labels")
        if not self.federated_data_dir:
            self.federated_data_dir = str(base / "federated/clients")
        if not self.global_model_dir:
            self.global_model_dir = str(base / "federated/global_models")
        if not self.client_models_dir:
            self.client_models_dir = str(base / "federated/client_models")

        # 创建所有目录
        for path in [
            self.processed_psg_dir, self.psg_labels_dir,
            self.federated_data_dir, self.global_model_dir, self.client_models_dir
        ]:
            os.makedirs(path, exist_ok=True)

    def __init__(self, **data):
        super().__init__(**data)
        self._init_paths()


class DatasetConfig(BaseModel):
    patient_ids: List[str] = []
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    client_data_split: str = "iid"
    non_iid_alpha: float = 0.5


class Config(BaseModel):
    project_name: str = "psg_to_sonar_federated"
    experiment_id: str = "v1.0"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42
    psg: PSGConfig = PSGConfig()
    generator: SonarGeneratorConfig = SonarGeneratorConfig()
    federated: FederatedConfig = FederatedConfig()
    paths: PathsConfig = PathsConfig()
    dataset: DatasetConfig = DatasetConfig()
    steps: List[str] = ["preprocess_psg", "split_federated_data", "federated_train"]


def load_config(config_file: str) -> Config:
    try:
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        return Config(**config_dict)
    except ValidationError as e:
        logging.error("配置文件验证失败:")
        logging.error(e)
        raise
    except FileNotFoundError:
        logging.error(f"配置文件不存在: {config_file}")
        raise


def generate_default_config(save_path: str = "psg_federated_config.yaml"):
    default = Config()
    with open(save_path, "w") as f:
        yaml.dump(default.dict(), f, sort_keys=False)
    logging.info(f"已生成默认配置文件: {save_path}")