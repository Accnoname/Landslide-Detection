import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset import LandslideDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Thiet bi: {device}")

    # encoder_weights=None vi chung ta nap truc tiep tu best_model.pth
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=14,
        classes=1,
        activation=None
    ).to(device)

    ckpt_path = "results/checkpoints/best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_dataset = LandslideDataset("data/processed/val/img", "data/processed/val/mask")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    all_probs = []
    all_targets = []

    print("[INFO] Dang thu thap xac suat du doan tren 245 mau Validation...")
    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            outputs = model(imgs.to(device))
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            targets = masks.numpy().flatten()
            all_probs.append(probs)
            all_targets.append(targets)

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    print("\n" + "="*65)
    print("=== BANG PHAN TICH ANH HUONG CUA NGUONG PHAN LOAI (THRESHOLD) ===")
    print("="*65)
    print(f"{'Threshold':^10} | {'Precision':^12} | {'Recall':^12} | {'F1-Score':^12} | {'IoU':^10}")
    print("-" * 65)

    thresholds = [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]
    best_th = 0.5
    best_f1 = 0.0
    results = []

    for th in thresholds:
        preds = (all_probs > th).astype(np.float32)
        tp = (preds * all_targets).sum()
        fp = (preds * (1 - all_targets)).sum()
        fn = ((1 - preds) * all_targets).sum()

        prec = tp / (tp + fp + 1e-7)
        rec = tp / (tp + fn + 1e-7)
        f1 = 2 * prec * rec / (prec + rec + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)

        if f1 > best_f1:
            best_f1 = f1
            best_th = th

        results.append((th, prec, rec, f1, iou))

    for th, prec, rec, f1, iou in results:
        marker = " <-- [TOI UU F1]" if th == best_th else ""
        print(f"   {th:.2f}    |   {prec*100:6.2f}%   |   {rec*100:6.2f}%   |   {f1*100:6.2f}%   |  {iou*100:5.2f}%{marker}")

    print("="*65)
    print(f"[KET LUAN] Nguong toi uu nhat la {best_th:.2f} dat F1 = {best_f1*100:.2f}%\n")

if __name__ == "__main__":
    main()
