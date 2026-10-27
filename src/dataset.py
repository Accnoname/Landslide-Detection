import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

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
    """
    def __init__(self, data_dir, mask_dir=None, transform=None):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        if self.mask_dir is not None and os.path.exists(self.mask_dir):
            self.image_files = sorted([
                f for f in os.listdir(data_dir)
                if f.endswith('.h5') and os.path.exists(os.path.join(self.mask_dir, f.replace('image_', 'mask_')))
            ])
        else:
            self.image_files = sorted([
                f for f in os.listdir(data_dir) if f.endswith('.h5')
            ])

        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # Đọc dữ liệu ảnh 14 channels
        with h5py.File(img_path, 'r') as hf:
            # Dữ liệu ảnh thường được lưu dưới key 'img' dạng (128, 128, 14)
            image = np.array(hf['img'], dtype=np.float32)
            
        # Chuẩn hóa đơn giản từng kênh về khoảng [0, 1] hoặc Z-score
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Đọc mask nhãn (nếu có)
        mask = None
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, img_name.replace('image_', 'mask_'))
            if os.path.exists(mask_path):
                with h5py.File(mask_path, 'r') as hf:
                    mask = np.array(hf['mask'], dtype=np.float32)
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Chuyển đổi định dạng sang PyTorch Tensor: (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image).float()
        
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
            return image_tensor, mask_tensor, img_name
            
        return image_tensor, img_name
