import os
import sys
import zipfile
import urllib.request
import shutil
import glob
from tqdm import tqdm

ZENODO_FILES = {
    "TrainData.zip": "https://zenodo.org/records/10463239/files/TrainData.zip?download=1",
    "ValidData.zip": "https://zenodo.org/records/10463239/files/ValidData.zip?download=1"
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_and_organize_dataset(target_data_dir="./data"):
    processed_dir = os.path.join(target_data_dir, "processed")
    train_img_dir = os.path.join(processed_dir, "train", "img")
    train_mask_dir = os.path.join(processed_dir, "train", "mask")
    val_img_dir = os.path.join(processed_dir, "val", "img")
    val_mask_dir = os.path.join(processed_dir, "val", "mask")

    # Kiểm tra nếu dữ liệu đã tồn tại đầy đủ thì bỏ qua
    if os.path.exists(train_img_dir) and len(os.listdir(train_img_dir)) > 100:
        print(f"✅ Dữ liệu huấn luyện đã tồn tại tại: {processed_dir} ({len(os.listdir(train_img_dir))} ảnh)")
        return

    os.makedirs(processed_dir, exist_ok=True)
    raw_dir = os.path.join(target_data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("="*70)
    print("🚀 Bắt đầu tải bộ dữ liệu Landslide4Sense từ Zenodo (Không cần đăng nhập)...")
    print("="*70)

    try:
        for fname, url in ZENODO_FILES.items():
            zip_dest = os.path.join(raw_dir, fname)
            if not os.path.exists(zip_dest) or os.path.getsize(zip_dest) < 1000000:
                print(f"\n[*] Đang tải {fname} ({url})...")
                download_url(url, zip_dest)
            
            print(f"[*] Đang giải nén {fname}...")
            with zipfile.ZipFile(zip_dest, 'r') as zip_ref:
                # Giải nén vào target
                if "Train" in fname:
                    target_split = os.path.join(processed_dir, "train")
                else:
                    target_split = os.path.join(processed_dir, "val")
                os.makedirs(target_split, exist_ok=True)
                zip_ref.extractall(target_split)

        # Chuẩn hóa cấu trúc thư mục nếu giải nén ra thư mục lồng
        for split in ["train", "val"]:
            split_dir = os.path.join(processed_dir, split)
            for sub in ["TrainData", "ValidData"]:
                nested = os.path.join(split_dir, sub)
                if os.path.exists(nested):
                    for item in os.listdir(nested):
                        src_item = os.path.join(nested, item)
                        dst_item = os.path.join(split_dir, item)
                        if not os.path.exists(dst_item):
                            shutil.move(src_item, dst_item)
                    try:
                        shutil.rmtree(nested)
                    except Exception:
                        pass

        print("\n" + "="*70)
        print("✅ TẢI VÀ GIẢI NÉN THÀNH CÔNG BỘ DỮ LIỆU LANDSLIDE4SENSE!")
        if os.path.exists(train_img_dir):
            print(f"📁 Train Images: {len(os.listdir(train_img_dir))} files")
        if os.path.exists(train_mask_dir):
            print(f"📁 Train Masks:  {len(os.listdir(train_mask_dir))} files")
        if os.path.exists(val_img_dir):
            print(f"📁 Val Images:   {len(os.listdir(val_img_dir))} files")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n[CẢNH BÁO] Không tải được từ Zenodo: {e}")
        print("[*] Chuyển sang phương án dự phòng tải từ Kagglehub...")
        try:
            import kagglehub
            cache_path = kagglehub.dataset_download("tekbahadurkshetri/landslide4sense")
            print(f"--> Kagglehub cache tại: {cache_path}")
            # Quét và copy
            for f in glob.glob(os.path.join(cache_path, "**", "*.h5"), recursive=True):
                fname = os.path.basename(f)
                if "mask" in f.lower() or fname.startswith("mask_"):
                    dest = os.path.join(train_mask_dir, fname)
                else:
                    dest = os.path.join(train_img_dir, fname)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if not os.path.exists(dest):
                    shutil.copy2(f, dest)
            print("✅ Tải dự phòng Kagglehub hoàn tất!")
        except Exception as e2:
            print(f"❌ Cả 2 phương án đều thất bại: {e2}")

if __name__ == "__main__":
    download_and_organize_dataset()
