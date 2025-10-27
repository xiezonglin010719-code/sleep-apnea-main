# src/federated/run_pipeline.py
import os
import torch
from torch import device

from src.federated import server
from src.federated.evaluate_generator import evaluate_generator
from src.federated.server import FederatedServer
from src.preprocessing.steps.configpsg import load_config

from src.models.generator_sonar import SonarFeatureGenerator

# run_pipeline.py 里新增
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.utils.metrics import plot_confusion_matrix, evaluate_classification


def build_global_val_loader(config, batch_size: int = 32, num_workers: int = 0):
    """
    合并 federated_data_dir/clients/*/val.npz 构建一个全局验证 DataLoader。
    兼容只有 features 而无 labels 的情况（用 -1 占位）。
    """
    root = "/Users/liyuxiang/Downloads/sleep-apnea-main/src/federated/data/clients"  # 例如 psg_federated/federated/clients
    if not os.path.isdir(root):
        raise FileNotFoundError(f"联邦数据目录不存在: {root}")

    feats_list, labels_list = [], []
    for name in sorted(os.listdir(root)):
        cdir = os.path.join(root, name)
        if not os.path.isdir(cdir):
            continue
        val_path = os.path.join(cdir, "val.npz")
        if not os.path.exists(val_path):
            # 没 val.npz 就跳过
            continue
        arr = np.load(val_path)
        if "features" not in arr.files:
            continue
        X = arr["features"]
        y = arr["labels"] if "labels" in arr.files else np.full((X.shape[0],), -1, dtype=np.int64)
        feats_list.append(X)
        labels_list.append(y)

    if not feats_list:
        raise RuntimeError(f"未在 {root} 下找到任何 val.npz")

    X_all = np.concatenate(feats_list, axis=0).astype(np.float32)
    y_all = np.concatenate(labels_list, axis=0).astype(np.int64)

    # 形状统一：支持 (N, D) 或 (N, T, F)。评估脚本会自己处理 forward 和 fake target。
    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.float32)  # 注意：评估里只把它当占位（或真值）；形状不匹配会自动改用假真值
    ds = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return loader

def build_test_loader(config, batch_size: int = 32, num_workers: int = 0):
    """
    合并 federated_data_dir/clients/*/test.npz 构建全局测试 DataLoader。
    若不存在独立测试集，尝试从验证集拆分一部分作为测试集。
    """
    root = config.paths.federated_data_dir
    if not os.path.isdir(root):
        raise FileNotFoundError(f"联邦数据目录不存在: {root}")

    feats_list, labels_list = [], []
    test_exists = False

    # 尝试加载独立测试集
    for name in sorted(os.listdir(root)):
        cdir = os.path.join(root, name)
        if not os.path.isdir(cdir):
            continue
        test_path = os.path.join(cdir, "test.npz")
        if os.path.exists(test_path):
            test_exists = True
            arr = np.load(test_path)
            if "features" in arr.files:
                X = arr["features"]
                y = arr["labels"] if "labels" in arr.files else np.full((X.shape[0],), -1, dtype=np.int64)
                feats_list.append(X)
                labels_list.append(y)

    # 若没有独立测试集，从验证集拆分
    if not test_exists:
        print("[WARNING] 未找到独立测试集，将从验证集拆分一部分作为测试集")
        val_loader = build_global_val_loader(config, batch_size=batch_size, num_workers=num_workers)
        # 转换为 numpy 数组
        X_val = np.concatenate([batch[0].numpy() for batch in val_loader], axis=0)
        y_val = np.concatenate([batch[1].numpy() for batch in val_loader], axis=0)
        # 按 50% 拆分验证集为 val 和 test
        split_idx = int(len(X_val) * 0.5)
        X_test, y_test = X_val[split_idx:], y_val[split_idx:]
        feats_list.append(X_test)
        labels_list.append(y_test)

    if not feats_list:
        raise RuntimeError(f"未在 {root} 下找到任何测试数据")

    X_all = np.concatenate(feats_list, axis=0).astype(np.float32)
    y_all = np.concatenate(labels_list, axis=0).astype(np.int64)

    # 构建数据集和加载器
    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.float32)
    ds = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return loader

def build_global_model(config, device):
    # 从配置取生成器参数（给默认值兜底）
    gen_cfg = getattr(config, "generator", None)
    input_dim  = int(getattr(gen_cfg, "input_dim", 256))     # 模型内部有自适配，36/91 也能跑
    output_dim = int(getattr(gen_cfg, "output_dim", 128))
    hidden     = list(getattr(gen_cfg, "hidden_layers", [512, 256]))
    dropout    = float(getattr(gen_cfg, "dropout_rate", 0.3))
    activation = str(getattr(gen_cfg, "activation", "relu"))
    num_classes= int(getattr(config, "num_classes", 3))      # 没配就默认 4 类

    model = SonarFeatureGenerator(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=hidden,
        dropout_rate=dropout,
        activation=activation,
        num_classes=num_classes,
        temporal_pool="mean",  # 支持 (B,T,F) 输入
    ).to(device)
    return model


# 修改 run_pipeline.py 中的 main 函数（最小侵入式改动）
def main(cfg_path: str):
    config = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    steps = list(getattr(config, "steps", []))
    print("steps from config =>", steps)

    # -------- ① PSG 原始数据预处理（可选）--------
    if "preprocess_psg" in steps:
        # 为兼容不同文件组织，做多路径导入尝试
        pre_cls = None
        try:
            from src.psg.preprocessor import PSGPreprocessor as _PP
            pre_cls = _PP
        except Exception:
            try:
                from src.psg.preprocessor import PSGPreprocessor as _PP
                pre_cls = _PP
            except Exception:
                try:
                    # 若你把类放在别处，这里再补一个路径即可
                    from src.psg.preprocessor import PSGPreprocessor as _PP
                    pre_cls = _PP
                except Exception as e:
                    raise ImportError(
                        "未能导入 PSGPreprocessor，请确认模块路径。"
                        "可将类文件放到 src/preprocessing/psg_preprocessor.py 并导出同名类。"
                    ) from e

        pre = pre_cls(config)
        patient_ids = [str(p) for p in getattr(config.dataset, "patient_ids", [])]
        if not patient_ids:
            raise ValueError("config.dataset.patient_ids 为空，请在配置中列出要处理的患者 ID 列表。")
        print(f"[STEP] preprocess_psg → patients={patient_ids}")
        pre.batch_process(patient_ids)
        print("✅ PSG 数据预处理完成（已写出 *_features.npz）")

    # -------- ② processed → federated/clients 划分（可选）--------
    if "split_federated" in steps:
        # 同样做稳健导入
        split_fn = None
        try:
            from src.federated.split_from_processed import split_from_processed as _split
            split_fn = _split
        except Exception as e:
            raise ImportError(
                "未能导入 split_from_processed，请确认模块路径 "
                "(应为 src/federated/split_from_processed.py 并导出 split_from_processed)。"
            ) from e

        print("[STEP] split_federated")
        split_fn(cfg_path)
        print("✅ 联邦数据划分完成（已写出 federated/clients/*/{train,val}.npz）")

    # -------- ③ 联邦训练（原有逻辑，保留）--------
    if "federated_train" in steps:
        global_model = build_global_model(config, device)
        server = FederatedServer(global_model=global_model, config=config, device=device)
        fed_dir = config.paths.federated_data_dir
        if not os.path.isdir(fed_dir):
            raise FileNotFoundError(f"联邦数据目录不存在: {fed_dir}")

        # 启动联邦训练（已包含评估和模型保存）
        server.train()

        # 训练完成后加载最佳模型做最终评估
        best_model_path = os.path.join(config.paths.global_model_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            print(f"\n加载最佳模型进行最终评估: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device)
            global_model.load_state_dict(checkpoint['model_state_dict'])

            # 构建测试集加载器（若无独立测试集将从验证集拆分）
            test_loader = build_test_loader(config, batch_size=32)

            cls_names = config.psg.class_names
            num_classes = config.psg.num_classes

            # 最终评估
            final_results = evaluate_classification(
                global_model,
                test_loader,
                device,
                cls_names
            )

            # 保存最终评估报告
            report_path = os.path.join(config.paths.global_model_dir, "final_classification_report.txt")
            with open(report_path, "w") as f:
                f.write(f"Best Model (Round {checkpoint['round']}, F1: {checkpoint['f1_score']:.4f})\n")
                f.write("Test Set Results:\n")
                f.write(f"Loss: {final_results['loss']:.4f}\n")
                f.write(f"Macro F1: {final_results['f1_macro']:.4f}\n")
                f.write(f"Micro F1: {final_results['f1_micro']:.4f}\n")
                f.write("\nClassification Report:\n")
                f.write(final_results['classification_report'])

            # 保存最终混淆矩阵
            cls_names = config.psg.class_names
            final_cm_path = os.path.join(config.paths.global_model_dir, "final_confusion_matrix.png")
            plot_confusion_matrix(
                final_results['y_true'],
                final_results['y_pred'],
                cls_names,
                final_cm_path
            )

            print(f"最终评估报告已保存至: {report_path}")
            print(f"最终混淆矩阵已保存至: {final_cm_path}")

    print("所有步骤执行完成。")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/Users/liyuxiang/Downloads/sleep-apnea-main/psg_federated_config.yaml",
        help="配置文件路径（默认使用本地 psg_federated_config.yaml）"
    )
    args = parser.parse_args()
    main(args.config)
