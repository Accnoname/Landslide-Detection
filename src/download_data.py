import os
import shutil
import glob

def download_and_organize_dataset(target_data_dir="./data"):
    # Nếu đang ở Windows và có ổ D thì lưu cache vào D:\ để không đầy ổ C
    if os.path.exists("D:\\"):
        os.environ["KAGGLEHUB_CACHE"] = "D:\\kagglehub_cache"
        os.makedirs("D:\\kagglehub_cache", exist_ok=True)

    """
    Tải tập dữ liệu Landslide4Sense từ Kaggle qua kagglehub
    và tổ chức vào cấu trúc thư mục chuẩn:
      data/
      ├── raw/
      └── processed/
          ├── train/ (img, mask)
          └── val/   (img, mask)
    """
    try:
        import kagglehub
    except ImportError:
        print("--> Thư viện 'kagglehub' chưa được cài đặt. Đang cài đặt tự động...")
        os.system("pip install kagglehub")
        import kagglehub

    print("==================================================")
    print("🚀 Bắt đầu tải Landslide4Sense qua kagglehub...")
    print("==================================================")
    
    # Tải dataset về thư mục cache của hệ thống
    cache_path = kagglehub.dataset_download("tekbahadurkshetri/landslide4sense")
    print(f"\n--> Tải thành công! Vị trí cache: {cache_path}")

    # Đường dẫn đích
    raw_dir = os.path.join(target_data_dir, "raw")
    processed_dir = os.path.join(target_data_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print("\n--> Đang quét cấu trúc file vừa tải...")
    all_files = glob.glob(os.path.join(cache_path, "**", "*"), recursive=True)
    h5_files = [f for f in all_files if f.endswith('.h5')]
    print(f"--> Tìm thấy tổng cộng {len(h5_files)} file .h5.")

    # Phân loại và sao chép/tạo liên kết vào thư mục processed
    train_img_dir = os.path.join(processed_dir, "train", "img")
    train_mask_dir = os.path.join(processed_dir, "train", "mask")
    val_img_dir = os.path.join(processed_dir, "val", "img")
    val_mask_dir = os.path.join(processed_dir, "val", "mask")
    test_img_dir = os.path.join(processed_dir, "test", "img")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    copied_counts = {"train_img": 0, "train_mask": 0, "val_img": 0, "val_mask": 0, "test_img": 0}

    for f in h5_files:
        f_lower = f.lower().replace("\\", "/")
        fname = os.path.basename(f)
        
        # Xác định file thuộc tập nào
        if "train" in f_lower or "training" in f_lower:
            if "mask" in f_lower or fname.startswith("mask_"):
                dest = os.path.join(train_mask_dir, fname)
                copied_counts["train_mask"] += 1
            else:
                dest = os.path.join(train_img_dir, fname)
                copied_counts["train_img"] += 1
        elif "val" in f_lower or "valid" in f_lower:
            if "mask" in f_lower or fname.startswith("mask_"):
                dest = os.path.join(val_mask_dir, fname)
                copied_counts["val_mask"] += 1
            else:
                dest = os.path.join(val_img_dir, fname)
                copied_counts["val_img"] += 1
        elif "test" in f_lower:
            dest = os.path.join(test_img_dir, fname)
            copied_counts["test_img"] += 1
        else:
            # Nếu để phẳng cùng thư mục
            if fname.startswith("mask_"):
                dest = os.path.join(train_mask_dir, fname)
                copied_counts["train_mask"] += 1
            else:
                dest = os.path.join(train_img_dir, fname)
                copied_counts["train_img"] += 1

        if not os.path.exists(dest):
            shutil.copy2(f, dest)

    print("\n================ TỔNG KẾT TỔ CHỨC DỮ LIỆU ================")
    print(f"📁 Train Images:      {len(os.listdir(train_img_dir))} file")
    print(f"📁 Train Masks:       {len(os.listdir(train_mask_dir))} file")
    print(f"📁 Validation Images: {len(os.listdir(val_img_dir))} file")
    print(f"📁 Validation Masks:  {len(os.listdir(val_mask_dir))} file")
    print(f"📁 Test Images:       {len(os.listdir(test_img_dir))} file")
    print("==========================================================")
    print("✅ Đã hoàn tất đưa dữ liệu vào cấu trúc chuẩn tại:", processed_dir)

if __name__ == "__main__":
    download_and_organize_dataset()
