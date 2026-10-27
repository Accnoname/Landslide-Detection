import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def visualize_landslide_sample(h5_image_path, h5_mask_path=None, save_output_path="sample_visualization.png"):
    """
    Trực quan hóa một mẫu ảnh sạt lở gồm:
    1. Ảnh màu thực tế (True Color RGB: B04, B03, B02)
    2. Ảnh màu giả hồng ngoại (False Color NIR: B08, B04, B03)
    3. Độ cao địa hình (DEM)
    4. Độ dốc sườn núi (Slope)
    5. Mặt nạ nhãn sạt lở (Ground Truth Mask)
    """
    if not os.path.exists(h5_image_path):
        print(f"Không tìm thấy file ảnh: {h5_image_path}")
        return

    with h5py.File(h5_image_path, 'r') as hf:
        img = np.array(hf['img'])  # Shape: (128, 128, 14)

    # Đọc nhãn nếu có
    mask = None
    if h5_mask_path and os.path.exists(h5_mask_path):
        with h5py.File(h5_mask_path, 'r') as hf:
            mask = np.array(hf['mask'])

    # 1. True Color RGB (Bands 4, 3, 2 tương ứng index 3, 2, 1)
    rgb = img[:, :, [3, 2, 1]]
    # Chuẩn hóa min-max để hiển thị ảnh đẹp mắt
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb) + 1e-6)

    # 2. False Color (Bands 8, 4, 3 tương ứng index 7, 3, 2)
    nrg = img[:, :, [7, 3, 2]]
    nrg = (nrg - np.min(nrg)) / (np.max(nrg) - np.min(nrg) + 1e-6)

    # 3. Slope & DEM (Band 12, 13)
    slope = img[:, :, 12]
    dem = img[:, :, 13]

    # Vẽ biểu đồ
    fig, axes = plt.subplots(1, 5 if mask is not None else 4, figsize=(20, 4))
    
    axes[0].imshow(rgb)
    axes[0].set_title("1. True Color (RGB)")
    axes[0].axis('off')

    axes[1].imshow(nrg)
    axes[1].set_title("2. False Color (NIR/Red/Green)")
    axes[1].axis('off')

    im_slope = axes[2].imshow(slope, cmap='viridis')
    axes[2].set_title("3. Slope (Độ dốc)")
    axes[2].axis('off')
    plt.colorbar(im_slope, ax=axes[2], fraction=0.046, pad=0.04)

    im_dem = axes[3].imshow(dem, cmap='terrain')
    axes[3].set_title("4. Elevation (DEM)")
    axes[3].axis('off')
    plt.colorbar(im_dem, ax=axes[3], fraction=0.046, pad=0.04)

    if mask is not None:
        axes[4].imshow(mask, cmap='hot')
        axes[4].set_title("5. Nhãn Sạt lở (Ground Truth)")
        axes[4].axis('off')

    plt.tight_layout()
    plt.savefig(save_output_path, dpi=200)
    print(f"[OK] Da luu anh truc quan hoa tai: {save_output_path}")


if __name__ == "__main__":
    img_sample = "./data/processed/train/img/image_1.h5"
    mask_sample = "./data/processed/train/mask/mask_1.h5"
    out_img = "./results/plots/sample_visualization.png"
    os.makedirs("./results/plots", exist_ok=True)
    
    print("[INFO] Dang trich xuat va truc quan hoa mau anh:", img_sample)
    visualize_landslide_sample(img_sample, mask_sample, out_img)


