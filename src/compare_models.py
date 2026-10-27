import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from dataset import LandslideDataset
from train import build_model

def evaluate_model(model, loader, device, threshold=0.5):
    """Tính toán toàn diện các metrics và đo lường độ trễ (latency/FPS)"""
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            t0 = time.time()
            outputs = model(imgs)
            total_time += (time.time() - t0)
            total_samples += imgs.size(0)

            preds = (torch.sigmoid(outputs) > threshold).float().view(-1)
            targets = masks.float().view(-1)

            total_tp += (preds * targets).sum().item()
            total_fp += (preds * (1 - targets)).sum().item()
            total_fn += ((1 - preds) * targets).sum().item()
            total_tn += ((1 - preds) * (1 - targets)).sum().item()

    precision = total_tp / (total_tp + total_fp + 1e-7)
    recall = total_tp / (total_tp + total_fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-7)
    fps = total_samples / (total_time + 1e-7)
    ms_per_patch = (total_time / total_samples) * 1000

    return {
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "iou": iou * 100,
        "fps": fps,
        "latency_ms": ms_per_patch
    }

def count_parameters(model):
    """Đếm tổng số tham số có thể huấn luyện (triệu tham số)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def plot_comparison_chart(results, save_path):
    """Vẽ biểu đồ cột so sánh trực quan các mô hình"""
    models = [r["name"] for r in results]
    metrics = ["f1", "recall", "precision", "iou"]
    metric_labels = ["F1-Score (%)", "Recall (%)", "Precision (%)", "IoU (%)"]
    colors = ["#2563eb", "#16a34a", "#d97706", "#9333ea"]

    x = np.arange(len(models))
    width = 0.18

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2.5, 1]})

    # Đồ thị 1: So sánh Metrics
    for i, (m, label, c) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [r[m] for r in results]
        offset = (i - 1.5) * width
        rects = ax1.bar(x + offset, vals, width, label=label, color=c, alpha=0.9)
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.axhline(y=71.65, color='red', linestyle='--', linewidth=1.5, label='Benchmark Bài báo (71.65%)')
    ax1.set_title("So sánh Hiệu năng Phát hiện Sạt lở giữa các Mô hình", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax1.set_ylabel("Tỷ lệ (%)", fontsize=11)
    ax1.set_ylim(40, 90)
    ax1.grid(True, linestyle=":", alpha=0.6, axis='y')
    ax1.legend(loc='lower right', framealpha=0.95)

    # Đồ thị 2: So sánh Tốc độ suy luận (FPS) & Kích thước tham số
    fps_vals = [r["fps"] for r in results]
    param_vals = [r["params_M"] for r in results]

    ax2.bar(x - 0.18, fps_vals, 0.35, label="Tốc độ (FPS)", color="#0891b2", alpha=0.85)
    ax2.bar(x + 0.18, param_vals, 0.35, label="Tham số (Triệu Params)", color="#ea580c", alpha=0.85)

    for i in range(len(models)):
        ax2.annotate(f'{fps_vals[i]:.0f}', xy=(x[i] - 0.18, fps_vals[i]), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
        ax2.annotate(f'{param_vals[i]:.1f}M', xy=(x[i] + 0.18, param_vals[i]), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

    ax2.set_title("Tốc độ (FPS) & Dung lượng Mô hình", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=":", alpha=0.6, axis='y')
    ax2.legend(loc='upper right', framealpha=0.95)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Đánh giá và so sánh toàn diện các mô hình phân đoạn sạt lở")
    parser.add_argument("--ckpt_dir", type=str, default="./results/checkpoints")
    parser.add_argument("--val_img_dir", type=str, default="./data/processed/val/img")
    parser.add_argument("--val_mask_dir", type=str, default="./data/processed/val/mask")
    parser.add_argument("--output_plot", type=str, default="./results/plots/model_comparison_benchmark.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = LandslideDataset(args.val_img_dir, args.val_mask_dir)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Quét danh sách checkpoint có trong thư mục
    candidate_models = [
        {"name": "ResU-Net (ResNet34)", "arch": "unet", "encoder": "resnet34", "ckpt": "unet_resnet34_best.pth", "fallback_ckpt": "best_model.pth"},
        {"name": "DeepLabV3+ (ResNet34)", "arch": "deeplabv3plus", "encoder": "resnet34", "ckpt": "deeplabv3plus_resnet34_best.pth"},
        {"name": "FPN (ResNet34)", "arch": "fpn", "encoder": "resnet34", "ckpt": "fpn_resnet34_best.pth"},
        {"name": "UNet++ (ResNet34)", "arch": "unetplusplus", "encoder": "resnet34", "ckpt": "unetplusplus_resnet34_best.pth"}
    ]

    evaluated_results = []

    print("\n" + "="*80)
    print("[BENCHMARK] BANG DOI CHIEU SO SANH HIEU NANG CAC KIEN TRUC MO HINH")
    print("="*80)

    for item in candidate_models:
        ckpt_path = os.path.join(args.ckpt_dir, item["ckpt"])
        if not os.path.exists(ckpt_path) and "fallback_ckpt" in item:
            ckpt_path = os.path.join(args.ckpt_dir, item["fallback_ckpt"])

        if not os.path.exists(ckpt_path):
            continue

        model = build_model(arch=item["arch"], encoder_name=item["encoder"], in_channels=14, num_classes=1).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        params_m = count_parameters(model)

        metrics = evaluate_model(model, val_loader, device, threshold=0.5)
        metrics["name"] = item["name"]
        metrics["params_M"] = params_m
        metrics["epoch"] = checkpoint.get("epoch", "N/A")
        evaluated_results.append(metrics)

    if not evaluated_results:
        print("[CANH BAO] Khong tim thay checkpoint nao trong thu muc:", args.ckpt_dir)
        return

    # In bang so sanh
    print(f"{'Mo hinh':<24} | {'Params':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'IoU':<8} | {'Toc do':<8}")
    print("-" * 88)
    for r in evaluated_results:
        print(f"{r['name']:<24} | {r['params_M']:5.1f}M  | {r['precision']:8.2f}%  | {r['recall']:8.2f}%  | {r['f1']:8.2f}%  | {r['iou']:6.2f}% | {r['fps']:5.1f} FPS")
    print("=" * 88)

    # Xuat do thi
    plot_comparison_chart(evaluated_results, args.output_plot)
    print(f"[SUCCESS] Da ve bieu do so sanh cac mo hinh tai: {args.output_plot}\n")

if __name__ == "__main__":
    main()
