import argparse
import os
import yaml
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from src.models.wgan_gp import WGANGP
from src.utils.utils import load_pickle_events, to_uint8_image


# -------------------------
# 工具函数（复用原有逻辑）
# -------------------------
def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def resolve_dir(p, anchors=[Path.cwd()]):
    """解析路径"""
    if not p:
        return ""
    P = Path(p)
    if P.is_absolute():
        return str(P)
    for a in anchors:
        cand = a / P
        if cand.exists():
            return str(cand.resolve())
    return str((anchors[0] / P).resolve())


def count_original_samples(data_dir, label_mapping):
    """统计原始数据中各类别的样本数（按事件级统计）"""
    class_counts = {v: 0 for v in label_mapping.values()}
    subject_events = {}

    for filename in os.listdir(data_dir):
        if filename.endswith(".pickle"):
            pickle_path = os.path.join(data_dir, filename)
            events = load_pickle_events(pickle_path)
            subject_id = filename.split('.')[0]

            event_list = []
            for ev in events:
                event_label = ev.label.lower()
                if event_label in label_mapping:
                    class_counts[label_mapping[event_label]] += 1
                    event_list.append((to_uint8_image(ev.signal), label_mapping[event_label]))

            if event_list:
                subject_events[subject_id] = event_list

    return class_counts, subject_events


# -------------------------
# 针对性增强核心逻辑
# -------------------------
def augment_target_classes(config):
    """根据配置增强目标类别样本"""
    # 解析配置
    anchors = [Path.cwd(), Path(__file__).resolve().parents[2]]
    raw_data_dir = resolve_dir(config['paths']['signals_path'], anchors)
    save_dir = resolve_dir(config['paths']['augmented_save_path'], anchors)
    wgan_path = resolve_dir(config['paths']['wgan_generator_path'], anchors)
    target_classes = config['data_augmentation']['target_classes']
    baseline_cls = config['data_augmentation']['balance_baseline']
    augment_method = config['data_augmentation']['augment_method']

    # 创建增强数据保存目录（避免覆盖原始数据）
    os.makedirs(save_dir, exist_ok=True)
    print(f"增强数据将保存到：{save_dir}")

    # 加载标签映射和原始样本统计
    label_mapping = config['label_mapping']
    class_counts, subject_events = count_original_samples(raw_data_dir, label_mapping)
    baseline_count = class_counts[baseline_cls]
    print(f"\n原始样本分布：")
    for name, idx in label_mapping.items():
        print(f"  {name}: {class_counts[idx]}")
    print(f"平衡基准（{[k for k, v in label_mapping.items() if v == baseline_cls][0]}）样本数：{baseline_count}")

    # 计算每个目标类别需要补充的样本数
    need_augment = {}
    for cls in target_classes:
        cls_name = [k for k, v in label_mapping.items() if v == cls][0]
        current_count = class_counts[cls]
        need = max(0, baseline_count - current_count)
        need_augment[cls] = need
        print(f"目标类别 {cls_name}（{cls}）需要补充：{need} 个样本")

    # 加载WGAN生成器（若使用WGAN增强）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wgan = None
    if augment_method == "wgan":
        wgan = WGANGP(input_dim=100, img_channels=1, device=device)
        wgan.generator.load_state_dict(torch.load(wgan_path, map_location=device, weights_only=True))
        wgan.generator.eval()
        print(f"成功加载WGAN生成器：{wgan_path}")

    # 生成增强样本并保存（按类别生成独立文件，避免污染原始数据）
    for cls, need in need_augment.items():
        if need <= 0:
            continue

        cls_name = [k for k, v in label_mapping.items() if v == cls][0]
        print(f"\n开始生成 {need} 个 {cls_name} 增强样本...")

        # 生成增强数据（以事件为单位，后续可复用原有滑动窗口逻辑）
        augmented_events = []
        with torch.no_grad():
            for i in tqdm(range(0, need, config['data_augmentation']['wgan_batch_size'])):
                batch_size = min(config['data_augmentation']['wgan_batch_size'], need - i)

                # WGAN生成单张图像
                z = torch.randn(batch_size, 100, device=device)
                fake_imgs = wgan.generator(z)  # (batch_size, 1, H, W)

                # 转换为uint8图像（与原始数据格式一致）
                fake_imgs = (fake_imgs + 1) / 2 * 255  # 从[-1,1]映射到[0,255]
                fake_imgs = fake_imgs.clamp(0, 255).cpu().numpy().astype(np.uint8)
                fake_imgs = np.squeeze(fake_imgs, axis=1)  # 移除通道维度

                # 构建增强事件（模拟原始ApneaEvent格式，仅保留必要字段）
                for img in fake_imgs:
                    # 这里根据原始ApneaEvent结构调整，确保后续加载兼容
                    augmented_event = {
                        'signal': img,
                        'label': cls_name,
                        'index': len(augmented_events),
                        'acq_number': f"aug_{cls_name}_{len(augmented_events)}",
                        'start': 0,
                        'end': img.shape[1]  # 假设时间维度为图像宽度
                    }
                    augmented_events.append(augmented_event)

        # 保存增强样本（按类别命名，方便后续加载）
        save_path = os.path.join(save_dir, f"aug_{cls_name}_{len(augmented_events)}.pickle")
        with open(save_path, 'wb') as f:
            pickle.dump(augmented_events, f)
        print(f"{cls_name} 增强样本保存完成：{save_path}（{len(augmented_events)} 个事件）")

    print(f"\n针对性增强完成！增强数据位于：{save_dir}")


# -------------------------
# 主函数
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据预处理阶段针对性增强目标类别样本")
    parser.add_argument("--config", default="../../config.yaml", help="配置文件路径")
    args = parser.parse_args()

    # 加载配置并执行增强
    config = load_config(args.config)
    if config['data_augmentation']['enable']:
        augment_target_classes(config)
    else:
        print("数据增强已禁用（config.data_augmentation.enable=False）")