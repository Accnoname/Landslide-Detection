import os
import json
import argparse
import glob
import h5py
import numpy as np
import tifffile
from tqdm import tqdm

CHANNEL_NAMES = {
    "0": "B01_Aerosols",
    "1": "B02_Blue",
    "2": "B03_Green",
    "3": "B04_Red",
    "4": "B05_RedEdge1",
    "5": "B06_RedEdge2",
    "6": "B07_RedEdge3",
    "7": "B08_NIR",
    "8": "B08A_NarrowNIR",
    "9": "B09_WaterVapour",
    "10": "B11_SWIR1",
    "11": "B12_SWIR2",
    "12": "ALOS_DEM",
    "13": "ALOS_Slope"
}

def create_dataset_json(output_path, num_train, num_test=0, file_ending=".tif"):
    """Tạo file metadata dataset.json chuẩn của nnU-Net v2"""
    dataset_info = {
        "channel_names": CHANNEL_NAMES,
        "labels": {
            "background": 0,
            "landslide": 1
        },
        "numTraining": num_train,
        "file_ending": file_ending,
        "dataset_name": "Landslide4Sense_14Bands",
        "description": "Multi-source satellite imagery (Sentinel-2 + ALOS PALSAR) for landslide segmentation",
        "reference": "IEEE Transactions on Geoscience and Remote Sensing (TGRS 2022)"
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=4)
    print(f"[OK] Da tao metadata chuan nnU-Net v2 tai: {output_path}")

def convert_dataset(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, output_raw_dir, max_samples=0):
    """
    Chuyển đổi dữ liệu .h5 sang cấu trúc thư mục của nnU-Net v2:
    - imagesTr: case_XXXX_0000.tif ... case_XXXX_0013.tif
    - labelsTr: case_XXXX.tif
    - imagesTs: case_XXXX_0000.tif ... case_XXXX_0013.tif
    - labelsTs: case_XXXX.tif
    """
    ds_dir = os.path.join(output_raw_dir, "Dataset001_Landslide")
    images_tr_dir = os.path.join(ds_dir, "imagesTr")
    labels_tr_dir = os.path.join(ds_dir, "labelsTr")
    images_ts_dir = os.path.join(ds_dir, "imagesTs")
    labels_ts_dir = os.path.join(ds_dir, "labelsTs")

    for d in [images_tr_dir, labels_tr_dir, images_ts_dir, labels_ts_dir]:
        os.makedirs(d, exist_ok=True)

    print("\n" + "="*75)
    print("[nnU-Net v2] TIEN TRINH CHUYEN DOI DU LIEU 14 KENH SANG CHUAN QUOC TE")
    print("="*75)
    print(f"[*] Thu muc dich: {ds_dir}")

    # 1. Chuyển đổi tập Train
    train_files = sorted(glob.glob(os.path.join(train_img_dir, "*.h5")))
    if max_samples > 0:
        train_files = train_files[:max_samples]
        print(f"[CHE DO TEST] Chay thu nghiem voi {max_samples} mau train...")
    else:
        print(f"[*] Chuyen doi toan bo {len(train_files)} mau train...")

    converted_train = 0
    for idx, img_path in enumerate(tqdm(train_files, desc="Converting Train")):
        case_id = f"landslide_{idx:05d}"
        mask_path = os.path.join(train_mask_dir, os.path.basename(img_path).replace("image_", "mask_"))
        if not os.path.exists(mask_path):
            continue

        with h5py.File(img_path, "r") as hf:
            img = np.array(hf["img"], dtype=np.float32) # (128, 128, 14)
        with h5py.File(mask_path, "r") as hf:
            mask = np.array(hf["mask"], dtype=np.uint8) # (128, 128)

        # Lưu 14 kênh ảnh
        for c in range(14):
            chan_data = img[:, :, c]
            out_chan_file = os.path.join(images_tr_dir, f"{case_id}_{c:04d}.tif")
            tifffile.imwrite(out_chan_file, chan_data)

        # Lưu mask nhãn
        out_mask_file = os.path.join(labels_tr_dir, f"{case_id}.tif")
        tifffile.imwrite(out_mask_file, mask)
        converted_train += 1

    # 2. Chuyển đổi tập Validation (imagesTs & labelsTs)
    val_files = sorted(glob.glob(os.path.join(val_img_dir, "*.h5")))
    val_samples = min(len(val_files), max(10, max_samples // 4)) if max_samples > 0 else len(val_files)
    val_files = val_files[:val_samples]

    print(f"[*] Chuyen doi {len(val_files)} mau validation...")
    converted_val = 0
    for idx, img_path in enumerate(tqdm(val_files, desc="Converting Val")):
        case_id = f"val_landslide_{idx:05d}"
        mask_path = os.path.join(val_mask_dir, os.path.basename(img_path).replace("image_", "mask_"))
        if not os.path.exists(mask_path):
            continue

        with h5py.File(img_path, "r") as hf:
            img = np.array(hf["img"], dtype=np.float32)
        with h5py.File(mask_path, "r") as hf:
            mask = np.array(hf["mask"], dtype=np.uint8)

        for c in range(14):
            chan_data = img[:, :, c]
            out_chan_file = os.path.join(images_ts_dir, f"{case_id}_{c:04d}.tif")
            tifffile.imwrite(out_chan_file, chan_data)

        out_mask_file = os.path.join(labels_ts_dir, f"{case_id}.tif")
        tifffile.imwrite(out_mask_file, mask)
        converted_val += 1

    # 3. Tạo metadata dataset.json
    json_path = os.path.join(ds_dir, "dataset.json")
    create_dataset_json(json_path, num_train=converted_train, num_test=converted_val, file_ending=".tif")

    print("\n" + "="*75)
    print(f"[HOAN TAT] Da chuyen doi thanh cong:")
    print(f"   - Train Cases: {converted_train} (Moi case gom 14 channels tif + 1 mask tif)")
    print(f"   - Val Cases:   {converted_val}")
    print(f"   - Metadata:    {json_path}")
    print("="*75 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Chuyển đổi dữ liệu Landslide4Sense sang chuẩn nnU-Net v2")
    parser.add_argument("--train_img", type=str, default="data/processed/train/img")
    parser.add_argument("--train_mask", type=str, default="data/processed/train/mask")
    parser.add_argument("--val_img", type=str, default="data/processed/val/img")
    parser.add_argument("--val_mask", type=str, default="data/processed/val/mask")
    parser.add_argument("--output_raw", type=str, default="data/nnUNet_raw")
    parser.add_argument("--max_samples", type=int, default=30, help="So mau chay thu nghiem (0 = tat ca)")
    args = parser.parse_args()

    convert_dataset(
        train_img_dir=args.train_img,
        train_mask_dir=args.train_mask,
        val_img_dir=args.val_img,
        val_mask_dir=args.val_mask,
        output_raw_dir=args.output_raw,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()
