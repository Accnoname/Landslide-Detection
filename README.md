# 🛰️ HỆ THỐNG PHÁT HIỆN SẠT LỞ ĐẤT QUA ẢNH VỆ TINH ĐA PHỔ
## (Landslide Detection using Deep Learning & Multi-Source Satellite Imagery)

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Google_Colab-T4_GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" />
  <img src="https://img.shields.io/badge/Benchmark-IEEE_TGRS_2022-00629B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

> **Tác giả:** [Accnoname](https://github.com/Accnoname)  
> **Mã nguồn chính thức:** [https://github.com/Accnoname/Landslide-Detection](https://github.com/Accnoname/Landslide-Detection)  
> **Bộ dữ liệu mục tiêu:** **Landslide4Sense Benchmark** (14 kênh: Sentinel-2 + ALOS PALSAR DEM & Độ dốc)  
> **Thành tựu cốt lõi:** Tái lập và **vượt xa bài báo chuẩn quốc tế IEEE TGRS 2022** ($F_1 = \mathbf{75.77\%}$ vs $71.65\%$, $\text{IoU} = \mathbf{60.99\%}$ vs $55.83\%$, $\text{Recall} = \mathbf{77.49\%}$ vs $66.80\%$).

---

## 📌 1. BẢNG TỔNG KẾT KẾT QUẢ THỰC NGHIỆM ĐỐI ĐẦU

Mô hình đã được thử nghiệm và đối chiếu công bằng trên tập dữ liệu kiểm định độc lập gồm 245 mẫu viễn thám $128 \times 128 \times 14$ kênh:

| Kiến trúc Mô hình | Số lượng Tham số | Precision | Recall (Bắt trúng) | F1-Score | IoU (Đè khít) | Tốc độ suy luận | Đánh giá & Vai trò |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 **ResU-Net (ResNet34)** | **$24.5\text{M}$** | **$74.21\%$** | **$77.49\%$** | **$75.77\%$** | **$60.99\%$** | $266.9\text{ FPS}$ | 🏆 **Top 1 Toàn diện — Sắc nét ranh giới** |
| 🥈 **DeepLabV3+ (ResNet34)** | **$22.5\text{M}$** | $66.42\%$ | $77.03\%$ | **$71.33\%$** | $55.44\%$ | **$644.8\text{ FPS}$** | ⚡ **Tốc độ nhanh nhất — Giám sát diện rộng** |
| 🥉 **ResU-Net + TTA (8x)** | $24.5\text{M}$ | **$77.58\%$** | $73.91\%$ | **$75.70\%$** | $60.90\%$ | $34.5\text{ FPS}$ | 🛡️ **Lọc sạch nhiễu viền — Precision kỷ lục** |
| 🎯 **Bài báo gốc *(IEEE TGRS)*** | — | $77.20\%$ | $66.80\%$ | **$71.65\%$** | $55.83\%$ | — | 📐 Chuẩn tham chiếu cơ sở |

> [!TIP]
> * **Tại sao ResU-Net chiến thắng?** Nhờ các đường truyền tắt đa tầng (**Skip Connections** ở các mức $64\times 64, 32\times 32, 16\times 16$), ResU-Net bảo toàn nguyên vẹn độ sắc nét của mép sạt lở địa hình ở độ phân giải gốc $128 \times 128$.
> * **Tại sao Recall của dự án đạt $>77\%$ (vượt bài báo gốc $+10.7\%$)?** Sử dụng hàm mất mát **ComboLoss ($0.6\text{ Dice} + 0.4\text{ BCE}$)** để phạt nặng lỗi bỏ sót pixel sạt lở (vốn chỉ chiếm $<3\%$ diện tích).
> * **Bản chất IoU $\approx 61\%$:** IoU và F1 tuân theo quy luật toán học $\text{IoU} = \frac{F_1}{2 - F_1}$. Mức IoU $61\%$ là đẳng cấp rất cao trong phân đoạn ảnh vệ tinh, mép viền ôm sát thực tế.

---

## 🚀 2. HƯỚNG DẪN CHẠY 1-CLICK TRÊN GOOGLE COLAB (GPU MIỄN PHÍ)

Bạn có thể chạy toàn bộ quy trình từ clone code, tải dữ liệu, huấn luyện đến lưu checkpoint trực tiếp trên Google Colab mà không cần cài đặt môi trường máy tính:

👉 **[Bấm vào đây để mở trực tiếp `run_on_colab.ipynb` trên Google Colab](https://colab.research.google.com/github/Accnoname/Landslide-Detection/blob/main/run_on_colab.ipynb)**

### Quy trình 4 bước tự động trên Colab:
1. **Kết nối GPU:** Vào menu `Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ Chọn `T4 GPU`.
2. **Mount Google Drive:** Tự động tạo thư mục lưu trữ `MyDrive/Detect_Results/`.
3. **Tự động tải dữ liệu tốc độ cao:** Script tự kéo dữ liệu từ máy chủ Zenodo với mạng 1Gbps trong ~1 phút (hoàn toàn không cần tài khoản Kaggle).
4. **Bấm Run All (`Ctrl + F9`):** Huấn luyện 20 epochs chỉ mất **~3 phút**. File trọng số tốt nhất `unet_resnet34_best.pth` và biểu đồ học tập tự động lưu an toàn về Google Drive!

---

## 💻 3. HƯỚNG DẪN SỬ DỤNG MÃ NGUỒN CỤC BỘ (LOCAL WORKSTATION)

### 3.1. Cài đặt môi trường
```bash
# Clone repository
git clone https://github.com/Accnoname/Landslide-Detection.git
cd Landslide-Detection

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 3.2. Tải và chuẩn bị dữ liệu (Nếu chưa có data)
```bash
# Tự động tải từ Zenodo và phân loại cấu trúc thư mục
python src/download_data.py
python src/create_val_split.py
```

### 3.3. Huấn luyện mô hình
* **Huấn luyện ResU-Net tiêu chuẩn (20 epochs):**
  ```bash
  python src/train.py --arch unet --encoder resnet34 --batch_size 16 --epochs 20
  ```
* **Huấn luyện với Data Augmentation & Nạp RAM Caching (Tăng tốc đọc đĩa):**
  ```bash
  python src/train.py --arch unet --encoder resnet34 --batch_size 32 --epochs 20 --augment --cache_data
  ```
* **Huấn luyện DeepLabV3+ đối chứng:**
  ```bash
  python src/train.py --arch deeplabv3plus --encoder resnet34 --batch_size 32 --epochs 20
  ```
* **Huấn luyện tiếp nối từ Checkpoint (Fine-tuning):**
  ```bash
  python src/train.py --resume results/checkpoints/unet_resnet34_best.pth --epochs 30 --lr 1e-5
  ```

### 3.4. Đánh giá & Tối ưu hóa chuyên sâu
* **Quét ngưỡng xác suất tối ưu (Threshold Tuning từ 0.20 đến 0.80):**
  ```bash
  python src/tune_threshold.py --ckpt_path results/checkpoints/unet_resnet34_best.pth
  ```
* **Đánh giá Test-Time Augmentation (TTA 8x) khử nhiễu viền:**
  ```bash
  python src/evaluate_tta.py --ckpt_path results/checkpoints/unet_resnet34_best.pth
  ```
* **So sánh đối đầu toàn diện các mô hình (Xuất bảng & Đồ thị cột):**
  ```bash
  python src/compare_models.py
  ```
* **Xuất ảnh đối chiếu 5 cột (RGB, Slope, Ground Truth, AI Mask, Error Map):**
  ```bash
  python src/evaluate_and_visualize.py
  ```
* **Suy luận 800 ảnh Test chưa gán nhãn và nén gói ZIP nộp bài:**
  ```bash
  python src/predict_test.py
  ```

---

## 🔬 4. BẢN CHẤT HỌC THUẬT & MIỀN DỮ LIỆU ĐỊA KỸ THUẬT

### 4.1. Cơ cấu 14 kênh đầu vào (Multi-Source Remote Sensing)
Dữ liệu kết hợp quang phổ phản xạ thụ động của vệ tinh quang học với mô hình số hóa độ cao chủ động của radar:
1. **Kênh quang phổ Sentinel-2 (B01 - B12):**
   * `B02 (Blue), B03 (Green), B04 (Red)`: Màu sắc tự nhiên (vết sạt lở lộ đất màu nâu đỏ).
   * `B08, B08A (Near Infrared - NIR)`: **Chỉ số diệp lục lá cây**. Khi sạt lở cuốn trôi thảm thực vật, kênh NIR sụt giảm đột ngột (tín hiệu nhận diện mạnh nhất).
   * `B11, B12 (Short-Wave Infrared - SWIR)`: Độ ẩm bùn đất và khoáng vật trơ trọi.
2. **Kênh địa hình ALOS PALSAR (B13 & B14):**
   * `Slope (Độ dốc)`: Sạt lở tuân theo trọng lực vật lý, tập trung chủ yếu ở sườn dốc $20^\circ - 55^\circ$. Vùng đồng bằng bằng phẳng ($< 10^\circ$) có nguy cơ sạt lở bằng 0.
   * `DEM (Độ cao)`: Phân bố theo đai cao địa tầng đứt gãy.

### 4.2. Ý nghĩa thực tế của 4 chỉ số đánh giá (Evaluation Metrics)
* **Precision ($74.21\%$):** Thước đo chống báo nhầm. Ngăn chặn việc điều động lực lượng cứu hộ và máy xúc tốn kém đến những nơi chỉ là ruộng cày hay bãi cát.
* **Recall ($77.49\%$):** Thước đo chống bỏ sót. Trực tiếp bảo vệ tính mạng con người, đảm bảo không bỏ lọt các ngôi làng hay cung đường bị đất đá chôn vùi.
* **F1-Score ($75.77\%$):** Trung bình điều hòa giữa Precision và Recall, đại diện cho năng lực tổng thể của mô hình.
* **IoU ($60.99\%$):** Tỷ lệ diện tích giao trên hợp ($\text{IoU} = \frac{TP}{TP + FP + FN}$). Đánh giá độ đè khít hình thái học phục vụ đo đạc diện tích sạt lở ($m^2$).

---

## 📂 5. CẤU TRÚC MÃ NGUỒN CHUẨN MỰC (PRODUCTION-READY)

```text
Landslide-Detection/
├── configs/
│   └── baseline_resunet.yaml            # Cấu hình siêu tham số tiêu chuẩn
├── data/                                # Dữ liệu nghiên cứu (được lọc bởi .gitignore)
│   ├── raw/                             # File nén TrainData.zip, ValidData.zip
│   └── processed/                       # Dữ liệu sạch 100% (Zero Data Leakage)
│       ├── train/ (img, mask: 3,554 cặp)
│       ├── val/   (img, mask: 245 cặp)
│       └── test/  (img: 800 ảnh)
├── results/
│   ├── checkpoints/                     # Trọng số tốt nhất (.pth)
│   ├── plots/                           # Biểu đồ Loss, F1, Threshold curve, Ảnh 5 cột
│   ├── predictions/                     # 800 mask dự đoán & file ZIP nộp bài
│   └── unet_resnet34_training_log.csv   # Nhật ký số liệu 20 epochs
├── src/
│   ├── dataset.py                       # PyTorch Dataset hỗ trợ RAM Caching & Augmentation
│   ├── train.py                         # Pipeline huấn luyện linh hoạt (CLI + GPU)
│   ├── tune_threshold.py                # Quét ngưỡng xác suất nhạy cảm (0.20 -> 0.80)
│   ├── evaluate_tta.py                  # Đo lường Test-Time Augmentation 8 hướng
│   ├── compare_models.py                # Đánh giá & vẽ đồ thị so sánh ResU-Net vs DeepLabV3+
│   ├── evaluate_and_visualize.py        # Xuất ảnh đối chiếu 5 cột thực tế
│   ├── predict_test.py                  # Suy luận 800 ảnh test & đóng gói ZIP
│   ├── download_data.py                 # Tự động tải dữ liệu trực tiếp từ Zenodo
│   ├── create_val_split.py              # Script phân tách tập train/val độc lập
│   └── utils/
│       ├── losses.py                    # ComboLoss (0.6 Dice + 0.4 BCE)
│       └── metrics.py                   # Tính Precision, Recall, F1, IoU
├── requirements.txt                     # Danh sách thư viện phụ thuộc
├── run_on_colab.ipynb                   # Bệ phóng 1-click trên Google Colab
├── RESEARCH_NOTES.md                    # Sổ tay nghiên cứu & luận điểm phản biện đề tài
├── ROADMAP.md                           # Bản đồ nghiên cứu chi tiết 5 giai đoạn
├── .gitignore                           # Bảo mật chống lộ dữ liệu lớn và AI secrets
└── README.md                            # Tài liệu tổng quan dự án
```

---

## 📜 6. TRÍCH DẪN & TÀI LIỆU THAM KHẢO

1. **Bài báo Benchmark gốc:**  
   *Ghorbanzadeh, O. et al. (2022). "Landslide4Sense: Reference Benchmark Data and Deep Learning Models for Landslide Detection." IEEE Transactions on Geoscience and Remote Sensing (TGRS).*
2. **Review tổng quan viễn thám:**  
   *MDPI Remote Sensing (2024). "Deep Learning for Landslide Detection from Satellite Imagery: Advances and Challenges."*
3. **nnU-Net Framework:**  
   *Isensee, F. et al. (2021). "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature Methods.*

---
*Dự án được thực hiện và công bố bởi tác giả **Accnoname**. Mọi ý kiến đóng góp hoặc câu hỏi xin vui lòng mở Issue trên GitHub repository.*
