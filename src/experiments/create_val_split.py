import os
import shutil
import random

def create_labeled_val_split(train_dir="./data/processed/train", val_dir="./data/processed/val", num_val=245, seed=42):
    """
    Trích xuất 245 mẫu có nhãn từ TrainData đưa vào ValData
    để làm tập kiểm định (Validation Set) chuẩn cho mô hình tính F1-Score.
    """
    random.seed(seed)
    train_img_dir = os.path.join(train_dir, "img")
    train_mask_dir = os.path.join(train_dir, "mask")
    val_img_dir = os.path.join(val_dir, "img")
    val_mask_dir = os.path.join(val_dir, "mask")

    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)

    if os.path.exists(val_img_dir) and len([f for f in os.listdir(val_img_dir) if f.endswith('.h5')]) >= 100:
        print(f"[INFO] Tập Validation đã có sẵn {len(os.listdir(val_img_dir))} ảnh. Bỏ qua bước trích xuất!")
        return

    # Lấy danh sách ảnh có kèm mask
    train_images = sorted([f for f in os.listdir(train_img_dir) if f.endswith('.h5')])
    
    # Chọn ngẫu nhiên num_val mẫu
    val_selected = random.sample(train_images, num_val)
    print(f"[INFO] Chon ngau nhien {len(val_selected)} mau co nhan de tao tap Validation...")

    moved_count = 0
    for img_name in val_selected:
        mask_name = img_name.replace("image_", "mask_")
        
        src_img = os.path.join(train_img_dir, img_name)
        src_mask = os.path.join(train_mask_dir, mask_name)
        
        dst_img = os.path.join(val_img_dir, img_name)
        dst_mask = os.path.join(val_mask_dir, mask_name)

        if os.path.exists(src_img) and os.path.exists(src_mask):
            shutil.move(src_img, dst_img)
            shutil.move(src_mask, dst_mask)
            moved_count += 1

    print(f"[OK] Da chuyen thanh cong {moved_count} cap (anh, nhan) vao thu muc Validation!")
    print(f"--> So luong tap Train hien tai: {len(os.listdir(train_img_dir))} anh")
    print(f"--> So luong tap Valid hien tai: {len(os.listdir(val_img_dir))} anh (co day du nhan de tinh F1)")

if __name__ == "__main__":
    create_labeled_val_split()
