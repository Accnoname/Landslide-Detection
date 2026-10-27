import os
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class LandslideDataset(Dataset):
    """
    Dataset loader cho Landslide4Sense (File .h5)
    
    Cấu trúc 14 channels đầu vào:
      - Bands 01 -> 12: Sentinel-2 Multispectral bands
          * B01: Aerosols (443nm)
          * B02: Blue (490nm)
          * B03: Green (560nm)
          * B04: Red (665nm)
          * B05-B07: Red Edge
          * B08: NIR (842nm)
          * B08A: Narrow NIR
          * B09: Water vapour
          * B11-B12: SWIR
      - Band 13: ALOS PALSAR Slope (Độ dốc địa hình)
      - Band 14: ALOS PALSAR Elevation (Độ cao địa hình - DEM)
      
    Label:
      - Binary Mask: 0 (Không sạt lở), 1 (Có sạt lở)

    Hỗ trợ:
      - augment: True/False (Tăng cường dữ liệu 2D: Xoay 90, 180, 270, Lật ngang/dọc)
      - cache_in_memory: True/False (Nạp toàn bộ dữ liệu vào RAM để tăng tốc đọc đĩa)
    """
    def __init__(self, data_dir, mask_dir=None, augment=False, cache_in_memory=False):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.cache_in_memory = cache_in_memory
        
        if self.mask_dir is not None and os.path.exists(self.mask_dir):
            self.image_files = sorted([
                f for f in os.listdir(data_dir)
                if f.endswith('.h5') and os.path.exists(os.path.join(self.mask_dir, f.replace('image_', 'mask_')))
            ])
        else:
            self.image_files = sorted([
                f for f in os.listdir(data_dir) if f.endswith('.h5')
            ])

        self.cached_images = []
        self.cached_masks = []

        if self.cache_in_memory and len(self.image_files) > 0:
            print(f"[CACHE] Đang nạp {len(self.image_files)} mẫu vào RAM để tăng tốc đọc dữ liệu...")
            for idx in tqdm(range(len(self.image_files)), desc="Caching to RAM", leave=False):
                img, mask = self._load_from_disk(idx)
                self.cached_images.append(img)
                self.cached_masks.append(mask)
            print("✅ Đã nạp dữ liệu hoàn tất vào RAM!")

    def __len__(self):
        return len(self.image_files)

    def _load_from_disk(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        with h5py.File(img_path, 'r') as hf:
            image = np.array(hf['img'], dtype=np.float32) # (128, 128, 14)
            
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        
        mask = None
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, img_name.replace('image_', 'mask_'))
            if os.path.exists(mask_path):
                with h5py.File(mask_path, 'r') as hf:
                    mask = np.array(hf['mask'], dtype=np.float32)
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        return image, mask

    def _apply_augmentation(self, image, mask):
        """
        Data Augmentation không gian 2D cho ảnh 14 kênh và nhãn:
        - Xoay ngẫu nhiên 90, 180, 270 độ
        - Lật ngẫu nhiên theo trục ngang và dọc
        (Bảo toàn nguyên vẹn giá trị phổ và địa hình của từng pixel)
        """
        # 1. Random Flip ngang
        if random.random() > 0.5:
            image = np.fliplr(image)
            if mask is not None:
                mask = np.fliplr(mask)
                
        # 2. Random Flip dọc
        if random.random() > 0.5:
            image = np.flipud(image)
            if mask is not None:
                mask = np.flipud(mask)
                
        # 3. Random Rotation (0, 90, 180, 270 độ)
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k, (0, 1))
            if mask is not None:
                mask = np.rot90(mask, k, (0, 1))
                
        return np.ascontiguousarray(image), np.ascontiguousarray(mask) if mask is not None else None

    def __getitem__(self, idx):
        if self.cache_in_memory:
            image = self.cached_images[idx].copy()
            mask = self.cached_masks[idx].copy() if self.cached_masks[idx] is not None else None
        else:
            image, mask = self._load_from_disk(idx)

        # Áp dụng Augmentation nếu bật chế độ huấn luyện
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # Chuyển đổi định dạng sang PyTorch Tensor: (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image).float()
        
        img_name = self.image_files[idx]
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
            return image_tensor, mask_tensor, img_name
            
        return image_tensor, img_name
