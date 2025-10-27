import os
import numpy as np
from pathlib import Path
from typing import List, Dict

from src.preprocessing.steps.configpsg import Config


class FederatedDataSplitter:
    def __init__(self, config: Config):
        self.config = config
        self.processed_psg_dir = config.paths.processed_psg_dir
        self.federated_dir = config.paths.federated_data_dir
        self.num_clients = config.federated.num_clients
        self.split_strategy = config.dataset.client_data_split
        self.alpha = config.dataset.non_iid_alpha

    def _load_patient_data(self, patient_id: str) -> Dict:
        data_path = os.path.join(self.processed_psg_dir, f"{patient_id}_features.npz")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"患者{patient_id}的特征文件不存在: {data_path}")
        data = np.load(data_path)
        return {"features": data["features"], "labels": data["labels"], "patient_id": patient_id}

    def _split_iid(self, all_data: List[Dict]) -> List[Dict]:
        all_features = np.concatenate([d["features"] for d in all_data], axis=0)
        all_labels = np.concatenate([d["labels"] for d in all_data], axis=0)
        shuffled_idx = np.random.permutation(len(all_features))
        features = all_features[shuffled_idx]
        labels = all_labels[shuffled_idx]

        per_client = len(features) // self.num_clients
        client_data = []
        for i in range(self.num_clients):
            start = i * per_client
            end = start + per_client if i < self.num_clients - 1 else len(features)
            client_data.append({"features": features[start:end], "labels": labels[start:end]})
        return client_data

    def _split_non_iid(self, all_data: List[Dict]) -> List[Dict]:
        client_data = [{"features": [], "labels": []} for _ in range(self.num_clients)]
        patient_assignments = np.random.choice(
            self.num_clients, size=len(all_data),
            p=[self.alpha ** i / sum(self.alpha ** j for j in range(self.num_clients)) for i in range(self.num_clients)]
        )
        for i, data in enumerate(all_data):
            client_idx = patient_assignments[i]
            client_data[client_idx]["features"].append(data["features"])
            client_data[client_idx]["labels"].append(data["labels"])

        for i in range(self.num_clients):
            client_data[i]["features"] = np.concatenate(client_data[i]["features"], axis=0)
            client_data[i]["labels"] = np.concatenate(client_data[i]["labels"], axis=0)
        return client_data

    def split_and_save(self, patient_ids: List[str]) -> None:
        all_data = [self._load_patient_data(pid) for pid in patient_ids]
        client_data = self._split_iid(all_data) if self.split_strategy == "iid" else self._split_non_iid(all_data)

        for i, data in enumerate(client_data):
            client_dir = os.path.join(self.federated_dir, f"client_{i:03d}")
            os.makedirs(client_dir, exist_ok=True)
            save_path = os.path.join(client_dir, "local_data.npz")
            np.savez_compressed(save_path, features=data["features"], labels=data["labels"])
            print(f"客户端{i}数据保存完成，样本数: {len(data['labels'])}")