import logging
import os
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import List, Union
import yaml


logging.basicConfig(level=logging.INFO)

class DataConfig(BaseModel):
    ids: List[str] = []
    targets: List[str] = ["ObstructiveApnea", "Hypopnea", "MixedApnea", "CentralApnea"]

class DownloadConfig(BaseModel):
    mode: str = "local"
    edf_urls: Union[List[str], None]
    rml_urls: Union[List[str], None]
    catalog_file: str
    bucket_name: Union[str, None]

class AudioConfig(BaseModel):
    data_channels: str = "Mic"
    edf_step_size: int
    sample_rate: int
    clip_length: int
    clip_overlap: int
    n_mels: int
    ids_to_process: Union[List[str], None] = None
    new_sample_rate: int = None
    image_size: int

# 在 existing imports 下方添加
class TrainingConfig(BaseModel):
    batch_size: int = 32  # 默认批量大小
    num_epochs: int = 20  # 训练轮数
    learning_rate: float = 0.001  # 学习率
    dropout: float = 0.3  # dropout比例
    num_classes: int = 2  # 分类数量（默认2类：无呼吸暂停/阻塞性呼吸暂停）
    val_split: float = 0.2  # 验证集比例
    test_split: float = 0.1  # 测试集比例


class PathsConfig(BaseModel):
    root: str = 'data'
    dataset_file_name: str = 'dataset.py'
    edf_download_path: str = Field(default=None)
    rml_download_path: str = Field(default=None)
    ambient_noise_path: str = Field(default=None)
    edf_preprocess_path: str = Field(default=None)
    rml_preprocess_path: str = Field(default=None)
    pickle_path: str = Field(default=None)
    audio_path: str = Field(default=None)
    spectrogram_path: str = Field(default=None)
    signals_path: str = Field(default=None)
    retired_path: str = Field(default=None)

    def _set_paths(self):
        base_path = Path(self.root)
        self.edf_download_path = os.path.join(base_path, 'downloads', 'edf')
        self.rml_download_path = os.path.join(base_path, 'downloads', 'rml')
        self.ambient_noise_path = os.path.join(base_path, 'downloads', 'ambient')
        self.edf_preprocess_path = os.path.join(base_path, 'preprocess', 'edf')
        self.rml_preprocess_path = os.path.join(base_path, 'preprocess', 'rml')
        self.pickle_path = os.path.join(base_path, 'preprocess', 'pickle')
        self.audio_path = os.path.join(base_path, 'processed', 'audio')
        self.spectrogram_path = os.path.join(base_path, 'processed', 'spectrogram')
        self.signals_path = os.path.join(base_path, 'processed', 'signals')
        self.retired_path = os.path.join(base_path, 'retired')

    def __init__(self, **data):
        super().__init__(**data)
        self._set_paths()

class Config(BaseModel):
    name: str
    version: str
    steps: List[str]
    data: DataConfig
    download: DownloadConfig
    audio: AudioConfig
    paths: PathsConfig
    training: TrainingConfig  # 新增训练配置

def load_config(config_file:str) -> Config:
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    try:
        return Config(**config)
    except ValidationError as e:
        logging.error("Pipeline config could not be initialized due to validation error:")
        logging.error(e)
        raise


