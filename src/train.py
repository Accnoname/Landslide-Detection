import os
import time
import csv
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import LandslideDataset
from utils.losses import ComboLoss
from utils.metrics import calculate_metrics

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

def build_model(arch="unet", encoder_name="resnet34", in_channels=14, num_classes=1):
    """Khởi tạo mô hình phân đoạn ngữ nghĩa theo kiến trúc được chọn"""
    if not HAS_SMP:
        raise ImportError("Vui lòng cài đặt segmentation_models_pytorch: pip install segmentation-models-pytorch")

    arch = arch.lower()
    if arch == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,      # Train từ đầu cho dữ liệu 14 kênh
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
    elif arch == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
    elif arch == "fpn":
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
    elif arch == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
    else:
        raise ValueError(f"Khong ho tro kien truc: {arch}. Cac lua chon: ['unet', 'deeplabv3plus', 'fpn', 'unetplusplus']")
    return model

def get_gpu_memory():
    """Lấy lượng VRAM đang sử dụng theo đơn vị GB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        return f"{allocated:.2f}GB / {reserved:.2f}GB"
    return "N/A (CPU)"

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_f1 = 0.0
    num_batches = len(loader)

    pbar = tqdm(enumerate(loader), total=num_batches, desc=f"[Epoch {epoch:02d}/{total_epochs:02d} Train]", leave=False)
    
    for step, (images, masks, _) in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Huấn luyện FP32 chuẩn xác tuyệt đối trên GPU (tránh tràn số NaN)
        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        # Gradient clipping giúp huấn luyện cực kỳ ổn định
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Tính metrics tức thời của batch
        with torch.no_grad():
            batch_metrics = calculate_metrics(outputs, masks)

        total_loss += loss.item()
        total_f1 += batch_metrics["f1"]

        current_lr = optimizer.param_groups[0]['lr']
        vram = get_gpu_memory()
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "F1": f"{batch_metrics['f1']*100:.1f}%",
            "Avg_F1": f"{(total_f1/(step+1))*100:.1f}%",
            "LR": f"{current_lr:.1e}",
            "GPU": vram
        })

    return total_loss / num_batches, total_f1 / num_batches

@torch.no_grad()
def validate_epoch(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    total_loss = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    num_batches = len(loader)

    pbar = tqdm(loader, desc=f"[Epoch {epoch:02d}/{total_epochs:02d} Valid]", leave=False)

    for images, masks, _ in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()

        preds = (torch.sigmoid(outputs) > 0.5).float().view(-1)
        targets = masks.float().view(-1)

        total_tp += (preds * targets).sum().item()
        total_fp += (preds * (1 - targets)).sum().item()
        total_fn += ((1 - preds) * targets).sum().item()

    avg_loss = total_loss / num_batches
    precision = total_tp / (total_tp + total_fp + 1e-7)
    recall = total_tp / (total_tp + total_fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-7)

    all_metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou
    }

    return avg_loss, all_metrics

def save_plots(history, save_path):
    """Tự động vẽ và cập nhật biểu đồ Loss & F1-Score"""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_f1 = [h["train_f1"] * 100 for h in history]
    val_f1 = [h["val_f1"] * 100 for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, 'b-o', label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_loss, 'r--s', label="Val Loss", linewidth=2)
    ax1.set_title("Biến thiên Hàm mất mát (Loss)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(epochs, train_f1, 'b-o', label="Train F1 (%)", linewidth=2)
    ax2.plot(epochs, val_f1, 'g-^', label="Val F1 (%)", linewidth=2)
    ax2.axhline(y=71.65, color='orange', linestyle=':', linewidth=2, label="Mục tiêu Bài báo (71.65%)")
    ax2.set_title("Chỉ số F1-Score (%) so với Bài báo", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1-Score (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện ResU-Net Phát hiện Sạt lở đất trên GPU")
    parser.add_argument("--train_img_dir", type=str, default="./data/processed/train/img")
    parser.add_argument("--train_mask_dir", type=str, default="./data/processed/train/mask")
    parser.add_argument("--val_img_dir", type=str, default="./data/processed/val/img")
    parser.add_argument("--val_mask_dir", type=str, default="./data/processed/val/mask")
    parser.add_argument("--output_dir", type=str, default="./results")
    
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--arch", type=str, default="unet", choices=["unet", "deeplabv3plus", "fpn", "unetplusplus"], help="Kien truc mo hinh: unet, deeplabv3plus, fpn, unetplusplus")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Backbone encoder (mac dinh: resnet34)")
    parser.add_argument("--resume", type=str, default="", help="Duong dan checkpoint (.pth) de tiep tuc huan luyen")
    parser.add_argument("--save_all", action="store_true", help="Luu rieng checkpoint o moi epoch")
    parser.add_argument("--augment", action="store_true", help="Bat tang cuong du lieu 2D (Data Augmentation) cho tap train")
    parser.add_argument("--cache_data", action="store_true", help="Nap du lieu vao RAM de tang toc huan luyen")

    args = parser.parse_args()
    model_tag = f"{args.arch}_{args.encoder}"
    
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*70)
    print(f"[INFO] KHOI DONG HUAN LUYEN MO HINH: {args.arch.upper()} (Backbone: {args.encoder})")
    print("="*70)
    print(f"[DEVICE] Thiet bi: {device} | Card: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"[MODEL]  Arch: {args.arch} | Encoder: {args.encoder} | Tag: {model_tag}")
    print(f"[VRAM]   Khoi diem: {get_gpu_memory()}")
    print(f"[CONFIG] Batch Size: {args.batch_size} | Epochs: {args.epochs} | Init LR: {args.lr}")
    print(f"[AUGMENT] Data Augmentation: {args.augment} | Cache RAM: {args.cache_data}")
    print("[MEO GPU] Xem GPU trong Task Manager: Tab Performance -> GPU -> doi '3D' thanh 'Cuda'")
    print("="*70)

    train_dataset = LandslideDataset(data_dir=args.train_img_dir, mask_dir=args.train_mask_dir, augment=args.augment, cache_in_memory=args.cache_data)
    val_dataset = LandslideDataset(data_dir=args.val_img_dir, mask_dir=args.val_mask_dir, augment=False, cache_in_memory=args.cache_data)

    # Kiem tra chong ro ri du lieu (Data Leakage Assertion)
    train_files = set([os.path.basename(f) for f in train_dataset.image_files])
    val_files = set([os.path.basename(f) for f in val_dataset.image_files])
    overlap = train_files.intersection(val_files)
    if len(overlap) > 0:
        raise ValueError(f"[CANH BAO] Phat hien {len(overlap)} mau bi ro ri giua Train va Val (Data Leakage)! Vui long tach tap du lieu doc lap.")

    print(f"[DATA] Du lieu Huan luyen: {len(train_dataset)} mau")
    print(f"[DATA] Du lieu Kiem thu:   {len(val_dataset)} mau")
    print(f"[DATA INTEGRITY] Kiem tra tap Train & Val: Doc lap 100% (Trung lap = {len(overlap)})")
    print("-"*70)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    model = build_model(arch=args.arch, encoder_name=args.encoder, in_channels=14, num_classes=1).to(device)
    criterion = ComboLoss(dice_weight=0.6, bce_weight=0.4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    best_f1 = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"[RESUME] Dang nap checkpoint tu: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_f1 = checkpoint.get("f1_score", 0.0)
        print(f"[RESUME] Nap thanh cong! Tiep tuc tu Epoch {start_epoch:02d} (Best F1 hien tai: {best_f1*100:.2f}%)")

    log_csv_path = os.path.join(args.output_dir, f"{model_tag}_training_log.csv")
    csv_mode = "a" if (args.resume and os.path.exists(log_csv_path)) else "w"
    csv_file = open(log_csv_path, mode=csv_mode, newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if csv_mode == "w":
        csv_writer.writerow(["epoch", "train_loss", "train_f1", "val_loss", "val_precision", "val_recall", "val_f1", "val_iou", "lr", "time_sec"])

    history = []
    start_total_time = time.time()

    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Train F1':^10} | {'Val Loss':^10} | {'Val Prec':^10} | {'Val Rec':^10} | {'Val F1':^10} | {'Val IoU':^10} | {'Status':^12}")
    print("-" * 105)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, epoch, args.epochs)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        elapsed = time.time() - t0

        val_f1 = val_metrics["f1"]
        val_prec = val_metrics["precision"]
        val_rec = val_metrics["recall"]
        val_iou = val_metrics["iou"]

        # Luon luu last model checkpoint
        last_ckpt_data = {
            "epoch": epoch,
            "arch": args.arch,
            "encoder": args.encoder,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "f1_score": val_f1,
            "metrics": val_metrics
        }
        torch.save(last_ckpt_data, os.path.join(ckpt_dir, f"{model_tag}_last.pth"))
        if model_tag == "unet_resnet34":
            torch.save(last_ckpt_data, os.path.join(ckpt_dir, "last_model.pth"))

        if args.save_all:
            torch.save(last_ckpt_data, os.path.join(ckpt_dir, f"{model_tag}_epoch_{epoch:02d}_f1_{val_f1*100:.1f}.pth"))

        status = ""
        if val_f1 > best_f1:
            diff = (val_f1 - best_f1) * 100
            best_f1 = val_f1
            status = f"[BEST +{diff:.1f}%]"
            best_ckpt_data = {
                "epoch": epoch,
                "arch": args.arch,
                "encoder": args.encoder,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "f1_score": best_f1,
                "metrics": val_metrics
            }
            torch.save(best_ckpt_data, os.path.join(ckpt_dir, f"{model_tag}_best.pth"))
            if model_tag == "unet_resnet34":
                torch.save(best_ckpt_data, os.path.join(ckpt_dir, "best_model.pth"))

        print(f"{epoch:02d}/{args.epochs:02d}  | {train_loss:^10.4f} | {train_f1*100:^9.2f}% | {val_loss:^10.4f} | {val_prec*100:^9.2f}% | {val_rec*100:^9.2f}% | {val_f1*100:^9.2f}% | {val_iou*100:^9.2f}% | {status:<12}")

        csv_writer.writerow([epoch, train_loss, train_f1, val_loss, val_prec, val_rec, val_f1, val_iou, current_lr, f"{elapsed:.1f}"])
        csv_file.flush()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_f1": val_f1
        })
        save_plots(history, os.path.join(plots_dir, f"{model_tag}_learning_curves.png"))
        if model_tag == "unet_resnet34":
            save_plots(history, os.path.join(plots_dir, "learning_curves.png"))

    csv_file.close()
    total_time = (time.time() - start_total_time) / 60
    print("=" * 105)
    print(f"[SUCCESS] HUAN LUYEN {args.arch.upper()} HOAN TAT trong {total_time:.1f} phut!")
    print(f"[TOP F1]  F1-Score cao nhat: {best_f1*100:.2f}% (Muc tieu bai bao: 71.65%)")
    print(f"[SAVE]    Checkpoint luu tai: {os.path.join(ckpt_dir, f'{model_tag}_best.pth')}")
    print(f"[PLOT]    Do thi tien trinh:  {os.path.join(plots_dir, f'{model_tag}_learning_curves.png')}")
    print("=" * 105)

if __name__ == "__main__":
    main()
