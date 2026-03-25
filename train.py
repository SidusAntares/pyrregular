"""
Corrected training script for Timematch dataset using MantisV2 as per official example.
This version strictly follows the provided example code structure.
"""

import argparse
import random
import sys

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils import data
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from tqdm import tqdm

# --- Imports for Timematch dataset ---
from dataset import PixelSetData, create_evaluation_loaders
from timematch_utils import label_utils
from timematch_utils.train_utils import bool_flag
from transforms import (
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    ToTensor,
    AddPixelLabels
)
from torchvision import transforms
import torch.nn.functional as F

# --- Global Variables ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M")


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune MantisV2 on Timematch dataset.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use (e.g., cuda:0, cpu). Auto-detected if not specified.')
    parser.add_argument('--per', default=0.3, type=float, help='Percentage of labeled samples to use for training/validation.')
    parser.add_argument('--seed', default=111, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers for data loading.')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for training.')
    parser.add_argument('--balance_source', type=bool_flag, default=True, help='Use class balanced batches for source.')
    parser.add_argument('--num_pixels', default=2, type=int, help='Number of pixels to sample from the input sample.')
    parser.add_argument('--seq_length', default=30, type=int, help='Number of time steps to sample from the input sample.')
    parser.add_argument('--data_root', default='/mnt/d/All_Documents/documents/ViT/dataset/timematch', type=str, help='Path to datasets root directory.')
    parser.add_argument('--source', default='france/30TXT/2017', type=str, help='Source domain.')
    parser.add_argument('--combine_spring_and_winter', action='store_true', help='Combine spring and winter classes.')
    parser.add_argument('--num_folds', default=1, type=int, help='Number of cross-validation folds.')
    parser.add_argument("--val_ratio", default=0.1, type=float, help='Validation ratio.')
    parser.add_argument("--test_ratio", default=0.2, type=float, help='Test ratio.')
    parser.add_argument('--sample_pixels_val', action='store_true', help='Sample pixels during validation.')
    parser.add_argument('--doy_p', action='store_true',help='doy project')
    # mantis
    parser.add_argument('--model', default = 'MultichannelProjector', type = str)
    parser.add_argument('--type', default = 'head', type = str)
    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.workers = args.num_workers
    return args


def get_data_loaders(splits, config, balance_source=True):
    """Creates and returns the training DataLoader."""
    strong_aug = transforms.Compose([
        RandomSamplePixels(config.num_pixels),
        RandomSampleTimeSteps(config.seq_length),
        Normalize(),
        ToTensor(),
        AddPixelLabels()
    ])

    source_dataset = PixelSetData(
        config.data_root, config.source, config.classes, strong_aug,
        indices=splits[config.source]['train'],
    )

    if balance_source:
        source_labels = source_dataset.get_labels()
        from collections import Counter
        freq = Counter(source_labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_labels]
        sampler = torch.utils.data.WeightedRandomSampler(source_weights, len(source_labels))
        print("Using balanced loader for source")
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            sampler=sampler,
            batch_size=config.batch_size,
            drop_last=True,
        )
    else:
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )
    print(f'Size of source dataset: {len(source_dataset)} ({len(source_loader)} batches)')
    return source_loader


def create_train_val_test_folds(datasets, num_folds, num_indices, val_ratio=0.1, test_ratio=0.2):
    """Creates train/val/test splits."""
    folds = []
    for _ in range(num_folds):
        splits = {}
        for dataset in datasets:
            if isinstance(num_indices, dict):
                indices = list(range(num_indices[dataset]))
            else:
                indices = list(range(num_indices))
            n = len(indices)
            n_test = int(test_ratio * n)
            n_val = int(val_ratio * n)
            n_train = n - n_test - n_val

            random.shuffle(indices)

            train_indices = set(indices[:n_train])
            val_indices = set(indices[n_train:n_train + n_val])
            test_indices = set(indices[-n_test:])
            assert set.intersection(train_indices, val_indices, test_indices) == set()
            assert len(train_indices) + len(val_indices) + len(test_indices) == n

            splits[dataset] = {'train': train_indices, 'val': val_indices, 'test': test_indices}
        folds.append(splits)
    return folds

def resize(X, size):
    X_scaled = F.interpolate(torch.tensor(X, dtype=torch.float), size=size, mode='linear', align_corners=False)
    return X_scaled.numpy()


def shape_adjust(batch_dict, doy_p=False, seq_len=32):
    """
    核心函数：将 batch_dict 转换为 MantisV2 所需的 (X, y) 格式。
    关键修复：在此处利用 'valid_pixels' 处理全无效时间步样本，防止 NaN。
    """
    pixels = batch_dict['pixels']  # [B, T, C, N]
    valid_pixels = batch_dict['valid_pixels']  # [B, T, N] - 注意这个维度！
    pixel_labels = batch_dict['pixel_labels']  # [B, N]

    B, T, C, N = pixels.shape

    # --- Step 1: 展平所有样本 ---
    # [B, T, C, N] -> [S, T, C] where S = B * N
    x_flat = pixels.permute(0, 3, 1, 2).reshape(-1, T, C)  # (S, T, C)
    y_flat = pixel_labels.reshape(-1)  # (S,)
    valid_flat = valid_pixels.permute(0, 2, 1).reshape(-1, T)  # (S, T)

    # --- Step 2: 【关键修复】处理全无效时间步样本 ---
    has_valid_time = valid_flat.any(dim=1)  # (S,)
    all_invalid_mask = ~has_valid_time  # (S,)

    if all_invalid_mask.any():
        count = all_invalid_mask.sum().item()
        print(f"⚠️ Fixing {count} all-invalid samples (forcing t=0 valid).")
        # 强制将第一个时间步标记为有效
        valid_flat[all_invalid_mask, 0] = 1.0

    # --- Step 3: 使用 valid_flat 清洗 x_flat ---
    # 策略：将无效时间步的数据替换为该像素在有效时间步上的均值
    x = x_flat.clone()
    for i in range(x_flat.shape[0]):
        valid_idx = valid_flat[i].bool()  # (T,)
        if valid_idx.any():
            # 计算每个通道在有效时间步上的均值 [C]
            mean_vals = x_flat[i, valid_idx, :].mean(dim=0, keepdim=True)  # (1, C)
            # 将无效时间步替换为均值
            x[i, ~valid_idx, :] = mean_vals
        else:
            # 理论上不会发生，因为上面已经修复了
            x[i] = 0.0



    # --- Step 5: 最终安全检查 ---
    x_np = x.cpu().numpy() if torch.is_tensor(x) else x
    if np.isnan(x_np).any() or np.isinf(x_np).any():
        print(f"DEBUG: Shape Adjust - Found NaN/Inf! Replacing with 0.")
        x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)

    y_np = y_flat.cpu().numpy() if torch.is_tensor(y_flat) else y_flat

    return x_np, y_np

def train(args):
    """Main training function."""
    print("=> Creating dataloader")
    config = args

    source_classes = label_utils.get_classes(
        config.source.split('/')[0], combine_spring_and_winter=config.combine_spring_and_winter
    )
    source_data = PixelSetData(config.data_root, config.source, source_classes)
    labels, counts = np.unique(source_data.get_labels(), return_counts=True)
    source_classes = [source_classes[i] for i in labels[counts >= 200]]
    print('Using classes:', source_classes)
    config.classes = source_classes
    config.num_classes = len(source_classes)

    # Control sample size
    total_num = len(source_data)
    if args.per == 1:
        use_num = total_num
        print(f"Using all {total_num} samples.")
    elif 0 <= args.per < 1:
        use_num = round(args.per * total_num)
        print(f"Limiting experiment pool to {use_num} random samples (Seed={args.seed}).")
    else:
        raise ValueError('Percentage must be between 0 and 1')
    print(f"(Seed={args.seed}).")

    # Create splits
    indices = {config.source: use_num}
    folds = create_train_val_test_folds([config.source], config.num_folds, indices, config.val_ratio, config.test_ratio)
    splits = folds[0]
    sample_pixels_val = config.sample_pixels_val
    val_loader, test_loader = create_evaluation_loaders(config.source, splits, config, sample_pixels_val)
    source_loader = get_data_loaders(splits, config, config.balance_source)

    print(f"Number of classes: {config.num_classes}")
    device = torch.device(args.device)


    # --- 5. Evaluate on Test Set ---
    print("=> Evaluating on test set...")
    try:
        y_pred = model.predict(x_test)

        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Macro F1-Score: {macro_f1:.4f}")

        # --- 6. Save Results ---
        source_name = 'AT1'
        match args.source:
            case 'france/30TXT/2017': source_name = 'FR1'
            case 'france/31TCJ/2017': source_name = 'FR2'
            case 'denmark/32VNH/2017': source_name = 'DK1'
            case _: source_name = 'AT1'

        file_name = f'./results/{source_name}/{args.model}_{args.fine_tuning_type}_{len(x_train)}_{timestamp}_Len{args.seq_len}_Seed{args.seed}.csv'
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)

        results_df = pd.DataFrame({
            'metric': ['accuracy', 'macro_f1'],
            'value': [accuracy, macro_f1]
        })
        results_df.to_csv(file_name, index=False)
        print(f"✅ Results saved to {file_name}")

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

    return model


def main():
    """Main execution function."""
    args = parse_args()
    seeds = [args.seed]
    print('Seeds to run:', seeds)

    for seed in seeds:
        print(f'--- Running with Seed = {seed} ---')
        # Set seed for this run
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        trained_model = train(args)
        print(f'Finished training with Seed = {seed}\n')


if __name__ == '__main__':
    main()