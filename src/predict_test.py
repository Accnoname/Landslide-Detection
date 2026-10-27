import os
import time
import zipfile
import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

class LandslideTestDataset(Dataset):
    """Dataset đọc 800 ảnh test (chưa có nhãn) từ file .h5"""
    def __init__(self, test_img_dir):
        self.test_img_dir = test_img_dir
        self.image_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.h5')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        fpath = os.path.join(self.test_img_dir, fname)

        with h5py.File(fpath, 'r') as hf:
            image = np.array(hf['img'], dtype=np.float32)

        # Xử lý giá trị bất thường
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        # Chuyển từ (128, 128, 14) sang (14, 128, 128)
        image = np.transpose(image, (2, 0, 1))

        return torch.from_numpy(image), fname

def normalize_rgb(img_np):
    """Chuẩn hóa kênh B04, B03, B02 sang RGB tự nhiên"""
    rgb = img_np[[3, 2, 1], :, :].transpose(1, 2, 0)
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
    return rgb

def visualize_test_predictions(sample_records, output_plot_path):
    """Vẽ 6 mẫu dự đoán tiêu biểu trên tập Test"""
    fig, axes = plt.subplots(len(sample_records), 3, figsize=(12, 3.5 * len(sample_records)))
    
    col_titles = ["1. Ảnh vệ tinh RGB (Sentinel-2)", "2. Độ dốc sườn núi (Slope)", "3. Vết sạt lở AI dự đoán"]

    for row_idx, (img_np, pred_mask, fname, num_pixels) in enumerate(sample_records):
        rgb = normalize_rgb(img_np)
        slope = img_np[12]

        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_ylabel(f"{fname}\n({num_pixels} px sạt lở)", fontsize=10, fontweight='bold')
        axes[row_idx, 1].imshow(slope, cmap='magma')
        axes[row_idx, 2].imshow(pred_mask, cmap='gray')

        for col_idx in range(3):
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_titles[col_idx], fontsize=12, fontweight='bold', pad=10)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=180)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Chạy suy luận sinh Mask dự đoán cho 800 ảnh Test")
    parser.add_argument("--test_img_dir", type=str, default="./data/processed/test/img")
    parser.add_argument("--checkpoint", type=str, default="./results/checkpoints/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="./results/predictions/masks")
    parser.add_argument("--threshold", type=float, default=0.50, help="Nguong quyet dinh xac suat pixel (mac dinh 0.50)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--create_zip", action="store_true", default=True, help="Nen toan bo 800 mask thanh file zip de nop bai")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*70)
    print("[RUN] BAT DAU TIEN TRINH SUY LUAN TOAN BO 800 ANH TAP TEST")
    print("="*70)
    print(f"[DEVICE]     Thiet bi su dung: {device}")
    print(f"[CHECKPOINT] Nap trong so:     {args.checkpoint}")
    print(f"[THRESHOLD]  Nguong phan loai: {args.threshold}")
    print(f"[DATA]       Thu muc anh Test: {args.test_img_dir}")
    print(f"[OUTPUT]     Thu muc xuat mask:{args.output_dir}")
    print("="*70)

    # 1. Nap mo hinh ResU-Net
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=14,
        classes=1,
        activation=None
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[INFO] Nap mo hinh thanh cong (Epoch {checkpoint.get('epoch', 'N/A')}, Best F1 = {checkpoint.get('f1_score', 0)*100:.2f}%)")

    # 2. Chuan bi DataLoader
    test_dataset = LandslideTestDataset(args.test_img_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"[INFO] Tim thay tong cong: {len(test_dataset)} anh test can du doan.\n")

    # 3. Chay suy luan tren GPU
    start_time = time.time()
    total_landslide_pixels = 0
    images_with_landslides = 0
    candidate_samples = []

    pbar = tqdm(test_loader, desc="[Inference 800 Test]", unit="batch")

    with torch.no_grad():
        for batch_images, batch_fnames in pbar:
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            probs = torch.sigmoid(outputs)
            preds = (probs > args.threshold).byte().cpu().numpy()

            for i, fname in enumerate(batch_fnames):
                pred_mask = preds[i, 0]  # Shape: (128, 128), uint8 (0 hoac 1)
                mask_pixels = int(np.sum(pred_mask))
                
                if mask_pixels > 0:
                    images_with_landslides += 1
                    total_landslide_pixels += mask_pixels

                # Luu mau tieu bieu de ve bieu do minh hoa
                if len(candidate_samples) < 20 and mask_pixels > 150:
                    candidate_samples.append((batch_images[i].cpu().numpy(), pred_mask, fname, mask_pixels))

                # Ghi file .h5 chuan cau truc ban to chuc
                mask_fname = fname.replace("image_", "mask_")
                output_h5_path = os.path.join(args.output_dir, mask_fname)

                with h5py.File(output_h5_path, 'w') as hf:
                    hf.create_dataset('mask', data=pred_mask, dtype='uint8')

    elapsed = time.time() - start_time
    speed = len(test_dataset) / elapsed

    print("\n" + "="*70)
    print("[SUCCESS] DA HOAN TAT DU DOAN TOAN BO 800 ANH TEST!")
    print("="*70)
    print(f"[*] Thoi gian thuc thi:      {elapsed:.2f} giay (~{speed:.1f} anh/giay)")
    print(f"[*] Tong so file mask da tao: {len(test_dataset)} file .h5")
    print(f"[*] So anh phat hien sat lo:  {images_with_landslides} / {len(test_dataset)} anh ({images_with_landslides/len(test_dataset)*100:.1f}%)")
    print(f"[*] Tong so pixel sat lo:     {total_landslide_pixels:,} pixels")
    area_sq_km = (total_landslide_pixels * 100) / 1e6
    print(f"[*] Tong dien tich sat lo:    ~{area_sq_km:.2f} km2 (do phan giai 10m x 10m)")
    print("="*70)

    # 4. Tao file ZIP tong hop
    zip_path = "./results/predictions/test_masks_submission.zip"
    if args.create_zip:
        print(f"\n[ZIP] Dang nen 800 file mask thanh: {zip_path}...")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(args.output_dir):
                for f in files:
                    if f.endswith('.h5'):
                        zipf.write(os.path.join(root, f), arcname=f)
        zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"[ZIP] Hoan tat nen file zip! Dung luong: {zip_size_mb:.2f} MB (San sang nop bai)")

    # 5. Ve anh truc quan hoa 6 mau du doan tieu bieu
    plot_path = "./results/plots/test_predictions_sample.png"
    if len(candidate_samples) >= 6:
        candidate_samples.sort(key=lambda x: x[3], reverse=True)
        visualize_test_predictions(candidate_samples[:6], plot_path)
        print(f"[PLOT] Da luu anh minh hoa 6 mau du doan Test tai: {plot_path}")

    print("="*70 + "\n")

if __name__ == "__main__":
    main()
