#!/usr/bin/env python3
"""
Model Evaluation and Metric Calculation Entrypoint.

Usage:
    # 1. Đánh giá chuẩn trên tập Validation
    python evaluate.py --checkpoint results/checkpoints/unet_resnet34_best.pth

    # 2. Đánh giá kèm quét ngưỡng quyết định tối ưu F1 (Threshold Tuning)
    python evaluate.py --checkpoint results/checkpoints/unet_resnet34_best.pth --tune_threshold

    # 3. Đánh giá kèm 8-fold Test-Time Augmentation (TTA)
    python evaluate.py --checkpoint results/checkpoints/unet_resnet34_best.pth --tta
"""
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure UTF-8 output on Windows consoles
for stream in [sys.stdout, sys.stderr]:
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8')
        except Exception:
            pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in [PROJECT_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data.dataset import LandslideDataset
from src.models.builder import build_model
from src.utils.metrics import calculate_metrics

def evaluate_checkpoint(checkpoint_path, val_img_dir, val_mask_dir, batch_size=16, threshold=0.5, tta=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"[EVALUATION] Bat dau danh gia mo hinh: {checkpoint_path}")
    print(f"[DEVICE]     Thiet bi: {device}")
    print(f"[SETTINGS]   Nguong: {threshold} | TTA: {tta}")
    print("=" * 70)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch = checkpoint.get("arch", "unet")
    encoder = checkpoint.get("encoder", "resnet34")

    model = build_model(arch=arch, encoder_name=encoder, in_channels=14, num_classes=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_dataset = LandslideDataset(data_dir=val_img_dir, mask_dir=val_mask_dir, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for images, masks, _ in tqdm(val_loader, desc="Evaluating", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            if tta:
                # 8-fold TTA (4 rotations x 2 flips)
                probs = torch.zeros_like(masks, dtype=torch.float32)
                for k in range(4):
                    for flip in [False, True]:
                        aug_img = torch.rot90(images, k, (2, 3))
                        if flip:
                            aug_img = torch.flip(aug_img, dims=[3])
                        
                        out = model(aug_img)
                        pred = torch.sigmoid(out)

                        if flip:
                            pred = torch.flip(pred, dims=[3])
                        pred = torch.rot90(pred, -k, (2, 3))
                        probs += pred
                probs /= 8.0
            else:
                outputs = model(images)
                probs = torch.sigmoid(outputs)

            preds = (probs > threshold).float()

            pred_flat = preds.view(-1)
            target_flat = masks.view(-1)

            total_tp += (pred_flat * target_flat).sum().item()
            total_fp += (pred_flat * (1.0 - target_flat)).sum().item()
            total_fn += ((1.0 - pred_flat) * target_flat).sum().item()

    precision = total_tp / (total_tp + total_fp + 1e-7)
    recall = total_tp / (total_tp + total_fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-7)

    print("\n" + "=" * 70)
    print(f"{'Metric':<25} | {'Value (%)':<15}")
    print("-" * 70)
    print(f"{'Precision':<25} | {precision * 100:6.2f}%")
    print(f"{'Recall':<25} | {recall * 100:6.2f}%")
    print(f"{'F1-Score':<25} | {f1 * 100:6.2f}%")
    print(f"{'IoU (Jaccard Index)':<25} | {iou * 100:6.2f}%")
    print("=" * 70)

    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}

def main():
    parser = argparse.ArgumentParser(description="Landslide Model Evaluation and Verification CLI")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pth checkpoint")
    parser.add_argument("--val_img_dir", type=str, default="./data/processed/val/img")
    parser.add_argument("--val_mask_dir", type=str, default="./data/processed/val/mask")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tta", action="store_true", help="Enable 8-fold Test-Time Augmentation")
    parser.add_argument("--tune_threshold", action="store_true", help="Scan optimal decision threshold from 0.20 to 0.80")

    args = parser.parse_args()

    if args.tune_threshold:
        from src.postprocess.tune_threshold import main as tune_main
        sys.argv = [sys.argv[0], "--ckpt_path", args.checkpoint, "--val_img_dir", args.val_img_dir, "--val_mask_dir", args.val_mask_dir]
        tune_main()
    else:
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            val_img_dir=args.val_img_dir,
            val_mask_dir=args.val_mask_dir,
            batch_size=args.batch_size,
            threshold=args.threshold,
            tta=args.tta
        )

if __name__ == "__main__":
    main()
