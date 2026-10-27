import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from dataset import LandslideDataset

def load_resunet(checkpoint_path, device):
    """Nạp mô hình ResU-Net với trọng số đã huấn luyện"""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=14,
        classes=1,
        activation=None
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Khong tim thay checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    epoch = checkpoint.get("epoch", "N/A")
    f1 = checkpoint.get("f1_score", 0.0)
    print(f"[INFO] Nap thanh cong checkpoint: {checkpoint_path}")
    print(f"[INFO] Mo hinh dat dinh tai Epoch: {epoch} | F1-Score: {f1*100:.2f}%\n")
    return model

def normalize_rgb(img_np):
    """Chuẩn hóa ảnh RGB (B04, B03, B02) để hiển thị tự nhiên"""
    # img_np shape: (14, 128, 128) -> RGB: [3, 2, 1]
    rgb = img_np[[3, 2, 1], :, :].transpose(1, 2, 0)
    # Clip 2% - 98% để tránh điểm quá chói hoặc quá tối
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
    return rgb

def create_error_overlay(gt_mask, pred_mask):
    """
    Tạo bản đồ phân tích đối chiếu lỗi:
    - Xanh lá (Green): True Positive (Mô hình nhận diện chính xác vết sạt lở)
    - Đỏ (Red): False Positive (Mô hình báo nhầm nơi không có sạt lở)
    - Vàng (Yellow): False Negative (Mô hình bỏ sót vết sạt lở thực tế)
    - Đen (Black): True Negative (Khu vực bình thường không sạt lở)
    """
    h, w = gt_mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    
    tp = (pred_mask == 1) & (gt_mask == 1)
    fp = (pred_mask == 1) & (gt_mask == 0)
    fn = (pred_mask == 0) & (gt_mask == 1)
    
    overlay[tp] = [0.0, 0.85, 0.15]   # Xanh lá: Đoán đúng
    overlay[fp] = [0.95, 0.15, 0.15]   # Đỏ: Báo nhầm
    overlay[fn] = [1.0, 0.85, 0.0]    # Vàng: Bỏ sót
    
    return overlay

def main():
    parser = argparse.ArgumentParser(description="Đánh giá & Trực quan hóa suy luận ResU-Net")
    parser.add_argument("--checkpoint", type=str, default="./results/checkpoints/best_model.pth")
    parser.add_argument("--val_img_dir", type=str, default="./data/processed/val/img")
    parser.add_argument("--val_mask_dir", type=str, default="./data/processed/val/mask")
    parser.add_argument("--output_path", type=str, default="./results/plots/inference_comparison.png")
    parser.add_argument("--num_samples", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("[RUN] BAT DAU TIEN TRINH TRUC QUAN HOA SUY LUAN RESU-NET")
    print("=" * 70)
    print(f"[DEVICE] Thiet bi su dung: {device}")

    model = load_resunet(args.checkpoint, device)
    val_dataset = LandslideDataset(data_dir=args.val_img_dir, mask_dir=args.val_mask_dir)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    # 1. Đánh giá toàn diện trên tập Validation (245 mẫu)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    print("[EVAL] Dang tinh toan metrics toan bo 245 mau Validation...")
    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            preds_flat = preds.view(-1)
            targets_flat = masks.view(-1)

            total_tp += (preds_flat * targets_flat).sum().item()
            total_fp += (preds_flat * (1 - targets_flat)).sum().item()
            total_fn += ((1 - preds_flat) * targets_flat).sum().item()
            total_tn += ((1 - preds_flat) * (1 - targets_flat)).sum().item()

    precision = total_tp / (total_tp + total_fp + 1e-7)
    recall = total_tp / (total_tp + total_fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-7)

    print("-" * 70)
    print(f"[*] KET QUA DANH GIA CHUAN XAC CUA BEST MODEL TREN TAP VALIDATION:")
    print(f"    - Precision: {precision*100:.2f}% (Muc tieu bai bao: 77.20%)")
    print(f"    - Recall:    {recall*100:.2f}% (Muc tieu bai bao: 66.80%)")
    print(f"    - F1-Score:  {f1*100:.2f}% (Muc tieu bai bao: 71.65%)")
    print(f"    - IoU:       {iou*100:.2f}% (Muc tieu bai bao: 55.83%)")
    print("-" * 70)

    # 2. Tìm các mẫu tiêu biểu có sạt lở để vẽ bảng so sánh trực quan
    sample_candidates = []
    for idx in range(len(val_dataset)):
        img_tensor, mask_tensor, fname = val_dataset[idx]
        landslide_pixels = mask_tensor.sum().item()
        if landslide_pixels > 200:  # Mẫu có vết sạt lở đáng kể
            sample_candidates.append((idx, landslide_pixels, fname))

    # Sắp xếp theo diện tích sạt lở đa dạng
    sample_candidates.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [c[0] for c in sample_candidates[:args.num_samples]]
    if len(selected_indices) < args.num_samples:
        selected_indices = list(range(args.num_samples))

    print(f"[PLOT] Dang trich xuat va ve {len(selected_indices)} mau sat lo tieu bieu...")

    # Tạo biểu đồ lưới: num_samples dòng x 5 cột
    fig, axes = plt.subplots(len(selected_indices), 5, figsize=(20, 4 * len(selected_indices)))
    
    col_titles = [
        "1. Ảnh màu thật (RGB)",
        "2. Độ dốc địa hình (Slope)",
        "3. Thực tế (Ground Truth)",
        "4. AI dự đoán (ResU-Net)",
        "5. Đối chiếu sai số (Error Map)"
    ]

    for row_idx, data_idx in enumerate(selected_indices):
        img_tensor, mask_tensor, fname = val_dataset[data_idx]
        img_input = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_input)
            prob = torch.sigmoid(output)[0, 0].cpu().numpy()
            pred_mask = (prob > 0.5).astype(np.uint8)

        img_np = img_tensor.numpy()
        gt_mask = mask_tensor[0].numpy().astype(np.uint8)
        
        # 1. RGB
        rgb = normalize_rgb(img_np)
        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_ylabel(f"Mẫu: {fname}", fontsize=11, fontweight='bold')
        
        # 2. Slope (Band 12)
        slope = img_np[12]
        axes[row_idx, 1].imshow(slope, cmap='magma')

        # 3. Ground Truth Mask
        axes[row_idx, 2].imshow(gt_mask, cmap='gray')

        # 4. Predicted Mask
        axes[row_idx, 3].imshow(pred_mask, cmap='gray')

        # 5. Error Map (Overlay)
        error_overlay = create_error_overlay(gt_mask, pred_mask)
        axes[row_idx, 4].imshow(error_overlay)

        for col_idx in range(5):
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_titles[col_idx], fontsize=13, fontweight='bold', pad=12)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    # Thêm chú thích cho cột Error Map ở phía dưới cùng
    fig.text(0.5, 0.01, 
             "[Chú thích Đối chiếu sai số]:  ■ Xanh lá: Nhận diện chính xác (TP)   |   ■ Đỏ: Báo nhầm (FP)   |   ■ Vàng: Bỏ sót vết sạt lở (FN)", 
             ha='center', fontsize=12, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#cccccc'))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, dpi=180)
    plt.close()

    print(f"[SUCCESS] Da xuat anh truc quan hoa thanh cong tai: {args.output_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
