import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from dataset import LandslideDataset

# Dam bao in tieng Viet tren terminal khong bi loi encoding
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="Quét ngưỡng xác suất tối ưu (Optimal Threshold Tuning)")
    parser.add_argument("--ckpt_path", type=str, default="results/checkpoints/unet_resnet34_best.pth",
                        help="Đường dẫn tới file checkpoint .pth")
    parser.add_argument("--val_img_dir", type=str, default="./data/processed/val/img",
                        help="Thư mục ảnh validation")
    parser.add_argument("--val_mask_dir", type=str, default="./data/processed/val/mask",
                        help="Thư mục nhãn validation")
    parser.add_argument("--output_plot", type=str, default="./results/plots/threshold_tuning_curve.png",
                        help="Đường dẫn lưu biểu đồ quét ngưỡng")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*70)
    print("[*] QUET NGUONG XAC SUAT TOI UU HOA F1-SCORE (THRESHOLD TUNING)")
    print("="*70)
    print(f"[DEVICE]     Thiet bi: {device}")
    
    # Dự phòng đường dẫn checkpoint
    ckpt_path = args.ckpt_path
    if not os.path.exists(ckpt_path):
        alt_paths = [
            "results/checkpoints/best_model.pth",
            "/content/drive/MyDrive/Detect_Results/checkpoints/unet_resnet34_best.pth"
        ]
        for p in alt_paths:
            if os.path.exists(p):
                ckpt_path = p
                break
    
    print(f"[CHECKPOINT] Dang nap: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Khong tim thay file checkpoint tai: {ckpt_path}")

    # Khởi tạo mô hình ResU-Net (ResNet34)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=14,
        classes=1,
        activation=None
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    val_dataset = LandslideDataset(args.val_img_dir, args.val_mask_dir)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == 'cuda'))

    all_probs = []
    all_targets = []

    print(f"[DATA]       Dang suy luan va thu thap xac suat tren {len(val_dataset)} mau Validation...")
    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            outputs = model(imgs.to(device))
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            targets = masks.numpy().flatten()
            all_probs.append(probs)
            all_targets.append(targets)

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    print("\n" + "="*70)
    print(f"{'Nguong (Th)':^12} | {'Precision':^12} | {'Recall':^12} | {'F1-Score':^12} | {'IoU':^10}")
    print("-" * 70)

    thresholds = [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    best_th = 0.5
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0
    best_iou = 0.0
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
            best_prec = prec
            best_rec = rec
            best_iou = iou

        results.append((th, prec, rec, f1, iou))

    for th, prec, rec, f1, iou in results:
        marker = " <-- [KY LUC F1]" if th == best_th else ""
        print(f"   {th:^6.2f}    |   {prec*100:6.2f}%   |   {rec*100:6.2f}%   |   {f1*100:6.2f}%   |  {iou*100:5.2f}%{marker}")

    print("="*70)
    print(f"[KET LUAN] Nguong toi uu nhat: {best_th:.2f}")
    print(f"           --> F1-Score dat:  {best_f1*100:.2f}% (Tang so voi nguong 0.50)")
    print(f"           --> Precision dat: {best_prec*100:.2f}%")
    print(f"           --> Recall dat:    {best_rec*100:.2f}%")
    print(f"           --> IoU dat:       {best_iou*100:.2f}%\n")

    # Vẽ biểu đồ quét ngưỡng
    os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)
    ths = [r[0] for r in results]
    precs = [r[1] * 100 for r in results]
    recs = [r[2] * 100 for r in results]
    f1s = [r[3] * 100 for r in results]
    ious = [r[4] * 100 for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(ths, f1s, 'b-o', linewidth=2.5, label=f"F1-Score (Dinh {best_f1*100:.2f}% tai {best_th})")
    plt.plot(ths, precs, 'g--s', linewidth=2, label="Precision (%)")
    plt.plot(ths, recs, 'r--^', linewidth=2, label="Recall (%)")
    plt.plot(ths, ious, 'm:', linewidth=2, label="IoU (%)")

    plt.axvline(x=best_th, color='black', linestyle='-.', alpha=0.7, label=f"Nguong toi uu Th={best_th}")
    plt.title("Phan tich Do nhay Nguong Quyet dinh (Threshold Sensitivity Curve)", fontsize=14, fontweight='bold')
    plt.xlabel("Nguong Quyet dinh (Decision Threshold)", fontsize=12)
    plt.ylabel("Gia tri Metric (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=200)
    plt.close()
    print(f"[PLOT] Da xuat bieu do phan tich nguong tai: {args.output_plot}\n")

if __name__ == "__main__":
    main()
