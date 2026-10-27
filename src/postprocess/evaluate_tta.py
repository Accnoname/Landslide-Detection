import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(project_root, "src")
for p in [project_root, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data.dataset import LandslideDataset
from src.models.builder import build_model

def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình với kỹ thuật Test-Time Augmentation (TTA 8x)")
    parser.add_argument("--ckpt_path", type=str, default="results/checkpoints/unet_resnet34_best.pth",
                        help="Đường dẫn tới file checkpoint .pth")
    parser.add_argument("--val_img_dir", type=str, default="./data/processed/val/img",
                        help="Thư mục ảnh validation")
    parser.add_argument("--val_mask_dir", type=str, default="./data/processed/val/mask",
                        help="Thư mục nhãn validation")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Ngưỡng phân loại xác suất")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()

def predict_single_batch_tta(model, imgs):
    """
    Test-Time Augmentation 8 hướng cho 1 batch ảnh:
    - 4 góc xoay: 0, 90, 180, 270 độ
    - 2 trạng thái lật: Gốc và Lật ngang
    Lấy trung bình xác suất của 8 phép biến đổi để triệt tiêu nhiễu.
    """
    batch_probs = torch.zeros((imgs.size(0), 1, imgs.size(2), imgs.size(3)), device=imgs.device)
    
    # 8 biến thể (4 rotations x 2 flips)
    variants = [
        (0, False), (1, False), (2, False), (3, False),
        (0, True),  (1, True),  (2, True),  (3, True),
    ]

    for k, flip in variants:
        x = imgs
        if flip:
            x = torch.flip(x, dims=[-1])
        if k > 0:
            x = torch.rot90(x, k, dims=[-2, -1])

        out = torch.sigmoid(model(x))

        # Đảo ngược lại về hình dạng ban đầu
        if k > 0:
            out = torch.rot90(out, -k, dims=[-2, -1])
        if flip:
            out = torch.flip(out, dims=[-1])

        batch_probs += out

    return batch_probs / len(variants)

def evaluate(model, loader, device, threshold, use_tta=False):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    desc = "Đánh giá TTA (8x)" if use_tta else "Đánh giá Chuẩn (1x)"
    with torch.no_grad():
        for imgs, masks, _ in tqdm(loader, desc=desc, leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device)

            if use_tta:
                probs = predict_single_batch_tta(model, imgs)
            else:
                probs = torch.sigmoid(model(imgs))

            preds = (probs > threshold).float().view(-1)
            targets = masks.float().view(-1)

            total_tp += (preds * targets).sum().item()
            total_fp += (preds * (1 - targets)).sum().item()
            total_fn += ((1 - preds) * targets).sum().item()

    prec = total_tp / (total_tp + total_fp + 1e-7)
    rec = total_tp / (total_tp + total_fn + 1e-7)
    f1 = 2 * prec * rec / (prec + rec + 1e-7)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-7)

    return {
        "precision": prec * 100,
        "recall": rec * 100,
        "f1": f1 * 100,
        "iou": iou * 100
    }

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*75)
    print("🚀 SO SÁNH HIỆU NĂNG: ĐÁNH GIÁ CHUẨN VS TEST-TIME AUGMENTATION (TTA 8x)")
    print("="*75)
    print(f"[DEVICE]     Thiết bị: {device}")
    print(f"[CHECKPOINT] {args.ckpt_path}")
    print(f"[THRESHOLD]  Ngưỡng phân loại: {args.threshold}")

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {args.ckpt_path}")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=14,
        classes=1,
        activation=None
    ).to(device)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    val_dataset = LandslideDataset(args.val_img_dir, args.val_mask_dir)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == 'cuda'))

    print(f"[DATA]       Số lượng mẫu Validation: {len(val_dataset)}")
    print("[*] Đang đo đạc chế độ Chuẩn (Single Pass)...")
    base_res = evaluate(model, val_loader, device, args.threshold, use_tta=False)

    print("[*] Đang đo đạc chế độ Test-Time Augmentation (8-Fold Averaging)...")
    tta_res = evaluate(model, val_loader, device, args.threshold, use_tta=True)

    print("\n" + "="*75)
    print(f"{'Phương pháp Đánh giá':<28} | {'Precision':^12} | {'Recall':^12} | {'F1-Score':^12} | {'IoU':^10}")
    print("-" * 75)
    print(f"{'1. Đánh giá Chuẩn (Không TTA)':<28} | {base_res['precision']:^10.2f}% | {base_res['recall']:^10.2f}% | {base_res['f1']:^10.2f}% | {base_res['iou']:^8.2f}%")
    print(f"{'2. Có TTA (8 Hướng Biến đổi)':<28} | {tta_res['precision']:^10.2f}% | {tta_res['recall']:^10.2f}% | {tta_res['f1']:^10.2f}% | {tta_res['iou']:^8.2f}%")
    print("-" * 75)
    
    diff_f1 = tta_res['f1'] - base_res['f1']
    diff_iou = tta_res['iou'] - base_res['iou']
    diff_prec = tta_res['precision'] - base_res['precision']
    diff_rec = tta_res['recall'] - base_res['recall']

    print(f"{'Mức độ Cải thiện (Delta)':<28} | {diff_prec:+10.2f}% | {diff_rec:+10.2f}% | {diff_f1:+10.2f}% | {diff_iou:+8.2f}%")
    print("="*75)
    if diff_f1 > 0:
        print(f"✅ TTA giúp tăng F1-Score thêm {diff_f1:+.2f}% mà KHÔNG CẦN HUẤN LUYỆN LẠI MÔ HÌNH!\n")
    else:
        print("✅ Kết quả hai chế độ tương đương nhau.\n")

if __name__ == "__main__":
    main()
