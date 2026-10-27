# HỆ THỐNG PHÁT HIỆN SẠT LỞ ĐẤT QUA ẢNH VỆ TINH ĐA PHỔ (LANDSLIDE4SENSE)
> **Tác giả:** Accnoname  
> **GitHub Repository:** [https://github.com/Accnoname/Landslide-Detection](https://github.com/Accnoname/Landslide-Detection)  
> **Tái lập & Nâng cao Benchmark Bài báo Quốc tế IEEE TGRS 2022**  
> Dữ liệu viễn thám đa nguồn: **Sentinel-2 (12 kênh quang phổ) + ALOS PALSAR (DEM & Slope)**

---

## 1. Tóm tắt Kết quả Nghiên cứu Thực nghiệm

Dự án đã tái lập và huấn luyện thành công các kiến trúc Deep Learning hàng đầu trên tập dữ liệu **Landslide4Sense** với kích thước $128 \times 128 \times 14$ kênh, đạt kết quả vượt mức cơ sở (baseline) của bài báo gốc:

| Kiến trúc Mô hình | Số lượng Tham số | Precision | Recall | F1-Score | IoU | Tốc độ suy luận (FPS) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **ResU-Net (ResNet34)** | **$24.5\text{M}$** | **$74.21\%$** | **$77.39\%$** | **$75.77\%$** | **$60.99\%$** | $266.9\text{ FPS}$ | 🏆 **Top 1 Toàn diện** |
| 🥈 **DeepLabV3+ (ResNet34)** | **$22.5\text{M}$** | $66.42\%$ | $77.03\%$ | **$71.33\%$** | $55.44\%$ | **$644.8\text{ FPS}$** | ⚡ **Tốc độ nhanh nhất** |
| 🎯 **Mục tiêu Bài báo *(IEEE TGRS)*** | — | $77.20\%$ | $66.80\%$ | **$71.65\%$** | $55.83\%$ | — | Chuẩn tham chiếu |

> [!TIP]
> * **ResU-Net** đạt độ chính xác cao nhất nhờ các đường truyền tắt (**Skip Connections**) đa tầng bảo toàn trọn vẹn ranh giới địa hình ở độ phân giải gốc $128 \times 128$.
> * **DeepLabV3+** có tốc độ xử lý nhanh gấp 2.4 lần, phù hợp cho bài toán giám sát diện rộng theo thời gian thực.
> * Cả hai mô hình đều được huấn luyện bằng hàm mất mát **ComboLoss ($0.6 \text{ Dice} + 0.4 \text{ BCE}$)** để kéo tỷ lệ tìm vết sạt lở (**Recall**) từ $66.80\%$ lên trên $77\%$.

---

## 2. Cấu trúc Dự án Sau khi Tối ưu & Dọn dẹp

Toàn bộ các thư mục rác và file tạm đã được làm sạch, giữ lại cấu trúc tinh gọn theo nguyên tắc Senior Dev (YAGNI):

```text
d:\Detect/
├── configs/
│   └── baseline_resunet.yaml            # Cấu hình siêu tham số tiêu chuẩn
├── data/
│   └── processed/                       # Dữ liệu sạch 100% (Zero Data Leakage)
│       ├── train/ (img, mask: 3,554 cặp)# Tập huấn luyện
│       ├── val/   (img, mask: 245 cặp)  # Tập kiểm định nội bộ độc lập
│       └── test/  (img: 800 ảnh)        # Tập kiểm thử chưa có nhãn
├── results/
│   ├── checkpoints/                     # Trọng số tốt nhất của từng mô hình (.pth)
│   ├── plots/                           # Toàn bộ biểu đồ học tập & ảnh trực quan hóa
│   ├── predictions/                     # 800 mask dự đoán & file nén ZIP nộp bài
│   ├── training_log.csv                 # Lịch sử ResU-Net
│   └── deeplabv3plus_resnet34_training_log.csv # Lịch sử DeepLabV3+
├── src/
│   ├── dataset.py                       # PyTorch Dataset đọc 14 channels từ .h5
│   ├── train.py                         # Pipeline huấn luyện đa mô hình trên GPU
│   ├── evaluate_and_visualize.py        # Trực quan hóa đối chiếu 5 cột thực tế
│   ├── tune_threshold.py                # Quét ngưỡng phân loại tối ưu (0.2 -> 0.8)
│   ├── compare_models.py                # Đánh giá & vẽ đồ thị so sánh ResU-Net vs DeepLabV3+
│   ├── predict_test.py                  # Suy luận 800 ảnh test & xuất gói ZIP nộp bài
│   ├── create_val_split.py              # Script phân tách tập train/val độc lập
│   ├── download_data.py                 # Tải dữ liệu từ Kaggle cache sang ổ D
│   ├── visualize_sample.py              # Trực quan hóa các kênh ảnh RGB, NIR, DEM, Slope
│   └── utils/
│       ├── losses.py                    # ComboLoss (0.6 Dice + 0.4 BCE)
│       └── metrics.py                   # Tính Precision, Recall, F1, IoU
├── requirements.txt                     # Danh sách thư viện phụ thuộc
├── ROADMAP.md                           # Bản đồ nghiên cứu chi tiết 5 giai đoạn
├── run_on_colab.ipynb                   # File Notebook chạy 1-click trên Google Colab
├── .gitignore                           # Bảo mật chống lộ dữ liệu lớn và AI secrets
└── README.md                            # Hướng dẫn tổng quan dự án
```

---

## 3. Hướng dẫn Lệnh Chạy Toàn diện (Cheat-Sheet)

### A. Huấn luyện Mô hình
* **Huấn luyện ResU-Net (20 epochs):**
  ```bash
  python src/train.py --arch unet --encoder resnet34 --batch_size 16 --epochs 20
  ```
* **Huấn luyện DeepLabV3+ (20 epochs):**
  ```bash
  python src/train.py --arch deeplabv3plus --encoder resnet34 --batch_size 16 --epochs 20
  ```
* **Huấn luyện tiếp nối từ checkpoint cũ (Resume):**
  ```bash
  python src/train.py --resume results/checkpoints/unet_resnet34_best.pth --epochs 40
  ```

### B. Đánh giá & So sánh Đối đầu
* **Chạy so sánh đối chiếu đa mô hình (Xuất bảng & Đồ thị cột):**
  ```bash
  python src/compare_models.py
  ```
* **Quét ngưỡng xác suất tối ưu (Threshold Tuning):**
  ```bash
  python src/tune_threshold.py
  ```

### C. Trực quan hóa & Xuất kết quả nộp bài
* **Xuất ảnh đối chiếu 5 cột (True Color, Slope, Ground Truth, AI Prediction, Error Map):**
  ```bash
  python src/evaluate_and_visualize.py
  ```
* **Chạy suy luận toàn bộ 800 ảnh Test & nén file ZIP:**
  ```bash
  python src/predict_test.py
  ```

---

## 4. Các Sản phẩm Minh chứng Sẵn sàng Nộp Báo cáo

1. **Gói ZIP nộp bài Test:** [test_masks_submission.zip](file:///d:/Detect/results/predictions/test_masks_submission.zip) (800 mask `.h5` chuẩn quốc tế).
2. **Biểu đồ so sánh mô hình:** [model_comparison_benchmark.png](file:///d:/Detect/results/plots/model_comparison_benchmark.png).
3. **Ảnh trực quan hóa 5 cột:** [inference_comparison.png](file:///d:/Detect/results/plots/inference_comparison.png).
4. **Biểu đồ hàm mất mát:** [unet_resnet34_learning_curves.png](file:///d:/Detect/results/plots/unet_resnet34_learning_curves.png).
