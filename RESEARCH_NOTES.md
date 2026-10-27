# SỔ TAY NGHIÊN CỨU & LUẬN ĐIỂM HỌC THUẬT ĐẮT GIÁ
## Dự án: Phát hiện Sạt lở đất qua Ảnh Vệ tinh Đa phổ (Landslide4Sense)
> **Tài liệu tổng hợp các phát hiện cốt lõi, tư duy địa kỹ thuật và luận điểm bảo vệ đề tài**  
> *Được đúc kết từ thực nghiệm mô hình (ResU-Net, DeepLabV3+) và 3 bài báo chuẩn quốc tế (MDPI 2024, China Geology 2024, IEEE TGRS).*

---

## 📌 PHẦN 1: BẢN CHẤT KỸ THUẬT & SO SÁNH HAI KIẾN TRÚC

### 1. Tại sao Precision của DeepLabV3+ lại thấp hơn ResU-Net?
* **Thực nghiệm trong dự án:**
  * **ResU-Net (ResNet34):** Precision đạt **74.21%**, Recall **77.39%**, F1 **75.77%**.
  * **DeepLabV3+ (ResNet34):** Precision chỉ đạt **66.42% ~ 70.38%**, trong khi Recall lại rất cao (**75.27% ~ 81.55%**).
* **4 Nguyên nhân cốt lõi:**
  1. **Kích thước ảnh nhỏ (128 x 128) xung đột cụm ASPP:** Ở tầng đáy (Bottleneck), feature map chỉ còn $8 \times 8$ (hoặc $4 \times 4$). ASPP áp dụng các bước nhảy giãn nở lớn (Dilation rates = 6, 12, 18) dẫn đến hiện tượng *"vượt biên"* và *Over-smoothing* (làm mịn quá mức), khiến vùng dự đoán bị phình to tròn (blob-like) thay vì ôm sát vết sạt lở.
  2. **Thiếu Skip Connections đa tầng:** U-Net nối tắt ở mọi tầng ($64\times 64, 32\times 32, 16\times 16, 8\times 8$), bảo toàn nguyên vẹn độ sắc nét viền răng cưa. DeepLabV3+ chỉ có 1 nhánh nông duy nhất, còn lại là nội suy phóng to (Bilinear Upsample 4x) từ tầng ngữ nghĩa trừu tượng -> viền bị nhòe sang đất lành -> **tăng False Positive (báo nhầm) -> kéo tụt Precision**.
  3. **Hàm mất mát ComboLoss (0.6 Dice + 0.4 BCE):** Trọng số Dice Loss 60% phạt rất nặng lỗi bỏ sót (False Negative), ép DeepLab "đoán mở rộng" để kéo Recall lên cao, dẫn đến viền lan ra ngoài.
  4. **Ngưỡng quyết định (Threshold 0.5):** DeepLab xuất ra nhiều pixel viền có xác suất $0.51 - 0.60$. Nâng ngưỡng lên $0.55 - 0.60$ sẽ giúp Precision tăng ngay 3% - 6%.

### 2. Bản chất của Skip Connection + Concat trong U-Net
* **Không phải là "cho xem đáp án":** Skip connection là bản sao của chính bức ảnh đề bài ở tầng nông (giàu chi tiết không gian, viền răng cưa nhưng hiểu ngữ nghĩa nông).
* **Cơ chế Concat (Ghép kênh):** Kẹp chồng 2 tập bản đồ cùng kích thước (ví dụ $16 \times 16$) theo chiều kênh ($128 + 128 = 256$ kênh). Kích thước $16 \times 16$ giữ nguyên, độ dày kênh tăng gấp đôi.
* **Tích hợp:** Lớp tích chập Conv 3x3 phía sau quét qua toàn bộ 256 kênh để kết hợp thông tin viền nét (từ Skip) với ngữ nghĩa nhận diện sạt lở (từ Decoder).

---

## 📌 PHẦN 2: BẢN CHẤT CỦA BÀI TOÁN PHÂN ĐOẠN NGỮ NGHĨA (SEGMENTATION)

### 1. Tại sao phải chuẩn xác đến "từng hạt pixel"?
* **Phân loại cấp độ Pixel (Pixel-level Classification):** Mỗi bức ảnh $128 \times 128 = 16.384$ pixel được gán nhãn 0 hoặc 1.
* **Ý nghĩa thực tế:** Mỗi pixel ảnh Sentinel-2 tương ứng **$10\text{m} \times 10\text{m} = 100\text{ m}^2$** ngoài thực địa.
  * Dự đoán sai lệch chỉ **10 pixel** = Sai số tới **$1.000\text{ m}^2$ đất**!
  * Sạt lở lấn thêm 2 - 3 pixel có thể đã chôn vùi một cung đường huyết mạch hoặc một cụm dân cư.
* **Chỉ số IoU (Intersection over Union):** Đo độ đè khít từng pixel giữa hình AI vẽ và hình chuyên gia vẽ. ResU-Net đạt IoU **60.99%** (vượt xa mức 55.83% của bài báo IEEE TGRS 2022).

---

## 📌 PHẦN 3: TƯ DUY ĐỘT PHÁ NÂNG CẤP BÀI TOÁN

### 1. Nâng cấp từ "Nhị phân" sang "Phân cấp độ nguy hiểm & Ưu tiên cứu hộ"
* **Hạn chế của bài toán nhị phân hiện tại:** Chỉ biết chỗ nào lở / không lở, không biết mức độ trầm trọng để điều phối cứu nạn.
* **Giải pháp 1: Multi-class Segmentation (4 phân vùng cứu hộ):**
  * *Lớp 0 (An toàn):* Rừng cây, đất lành.
  * *Lớp 1 (Vùng nguồn Scarp):* Vách đứt gãy đỉnh núi -> **Nguy cơ sạt lở tiếp diễn**, cấm cứu hộ tiếp cận từ trên cao.
  * *Lớp 2 (Vùng trượt Runout):* Dải đất đá trôi dạt.
  * *Lớp 3 (Vùng bồi lấp Accumulation):* Chân đồi nơi đất đá dồn ứ -> **Nơi chôn vùi nhà cửa, ưu tiên điều máy xúc cứu nạn số 1**.
* **Giải pháp 2: Bản đồ nhiệt nguy cơ (Heatmap Risk):**
  * Tận dụng xác suất liên tục $0.0 - 1.0$ từ hàm Sigmoid của mô hình hiện tại:
    * $0.0 - 0.2$ (Xanh lục): An toàn.
    * $0.2 - 0.5$ (Vàng): Vùng đệm nguy cơ / đất yếu.
    * $0.5 - 0.8$ (Cam): Sạt lở mức độ trung bình.
    * $0.8 - 1.0$ (Đỏ rực): Tâm chấn sạt lở nghiêm trọng nhất.
* **Giải pháp 3: Tích hợp GIS dân cư & hạ tầng:**
  $$\text{Chỉ số khẩn cấp cứu hộ} = \text{Mặt nạ AI} \times \text{Bản đồ mật độ dân cư \& Đường giao thông}$$

---

## 📌 PHẦN 4: NGHỊCH LÝ & NÚT THẮT CỦA AI VIỄN THÁM NĂM 2026
*(Đúc kết từ 3 bài báo: MDPI 8/2024, China Geology 2024, IEEE TGRS Landslide4Sense)*

1. **AI mượn lại công cụ cũ từ ảnh tự nhiên:** Các mạng CNN (U-Net, DeepLab) sinh ra để nhận diện chó, mèo, xe hơi (3 kênh RGB), coi 14 kênh vệ tinh như ma trận số phẳng lì, không hiểu quy luật địa chất vật lý.
2. **Cắt nhỏ ảnh (128 x 128) làm mất ngữ cảnh toàn cục:** Một sườn núi dài hàng cây số bị cắt vụn thành nhiều patch nhỏ, mô hình mất liên kết không gian sườn dốc lớn.
3. **Hiện tượng "Cùng phổ khác vật" (Spectral Confusion):** Đất nông nghiệp cày xới, bãi bồi ven sông, taluy đường đèo có màu và độ dốc rất giống sạt lở mới -> Nguyên nhân chính sinh ra False Positive.
4. **Nút thắt chuyển giao miền (Domain Shift):** Mô hình huấn luyện ở Nhật Bản, Đài Loan đem sang vùng nhiệt đới Việt Nam (Yên Bái, Lào Cai) sẽ sụt giảm độ chính xác do thổ nhưỡng đất đỏ bazan và thảm thực vật khác biệt hoàn toàn.
5. **Dự đoán từ ảnh tĩnh đơn lẻ thay vì chuỗi biến động (Change Detection):** Cần so sánh ảnh Trước vs Sau thảm họa để loại bỏ sai số.

---

## 📌 PHẦN 5: BÀI TOÁN "LÀM DÀY" KÊNH & GIẢI QUYẾT ĐỘ TRỄ SẠT LỞ (TIME-LAG)

### 1. Cơ cấu 14 kênh hiện tại & Khả năng mở rộng
* **Hiện tại (14 kênh):** 12 kênh quang phổ Sentinel-2 (B01-B12) + 2 kênh địa hình ALOS PALSAR (DEM + Slope).
* **Mở rộng (in_channels = 17+):** Hoàn toàn có thể nhúng thêm Lượng mưa, Lịch sử sạt lở, Thạch học, Khoảng cách đến đới đứt gãy đứt gãy địa chất.

### 2. Nghịch lý: "Mưa 3 ngày chưa sạt, nhưng 12 ngày sau mới sạt lở!"
* **Cơ chế địa chất:** Nước mưa ngấm từ từ xuống tầng sâu $5\text{m} - 20\text{m}$. Sau 10 - 12 ngày, nước tích tụ đầy mặt trượt ngầm làm bùng nổ **Áp lực nước lỗ rỗng (Pore Water Pressure)**, làm mất lực ma sát và sụp đổ quả núi kể cả khi trời đã tạnh mưa.
* **4 Giải pháp công nghệ AI để khắc phục:**
  1. **Chỉ số Mưa tích lũy suy giảm (Antecedent Precipitation Index - API trong 15-30 ngày):**
     $$\text{API} = \text{Mưa}_{t} + k \cdot \text{Mưa}_{t-1} + k^2 \cdot \text{Mưa}_{t-2} + \dots + k^{12} \cdot \text{Mưa}_{t-12} \quad (k \approx 0.85 - 0.90)$$
     Dù ngày 12 trời nắng, giá trị API vẫn ở mức báo động đỏ.
  2. **Mô hình Không - Thời gian (ConvLSTM / Spatio-Temporal Transformer):** Nhận chuỗi ảnh 15 ngày liên tiếp; CNN bắt đặc trưng địa hình, LSTM nhớ quá trình ngấm nước qua thời gian.
  3. **Đo trực tiếp độ sũng nước bằng Radar SAR (Sentinel-1):** Sóng radar xuyên qua mây đo hằng số điện môi của đất (Soil Moisture). Nhìn thấy rõ lòng đất sũng nước sau 12 ngày.
  4. **Đường cong Ngưỡng mưa nguy hiểm (Rainfall Intensity - Duration Threshold):** Mối quan hệ giữa cường độ mưa và thời gian dầm dề.

---

## 📌 PHẦN 6: BỘ CÂU HỎI & ĐÁP ÁN BẢO VỆ ĐỀ TÀI VỚI HỘI ĐỒNG / GIẢNG VIÊN

| Câu hỏi của Thầy / Hội đồng | Luận điểm trả lời ăn điểm tuyệt đối |
| :--- | :--- |
| **"Tại sao em chọn ResU-Net làm mô hình chính thay vì DeepLabV3+?"** | *"DeepLabV3+ chạy nhanh hơn (415 FPS so với 196 FPS) nhưng bị hạn chế ở ASPP trên patch nhỏ $128 \times 128$ và thiếu skip connections đa tầng, làm Precision chỉ đạt ~70.4%. Trong khi ResU-Net bảo toàn trọn vẹn ranh giới mép sạt lở sắc nét, đạt Precision 74.21% và F1 75.77%, vượt chuẩn bài báo IEEE TGRS."* |
| **"Hàm loss của em giải quyết mất cân bằng lớp như thế nào?"** | *"Vì vết sạt lở chỉ chiếm $<3\%$ diện tích ảnh, em kết hợp ComboLoss (0.6 Dice + 0.4 BCE). Dice Loss tập trung tối ưu hóa vùng giao nhau và phạt nặng việc bỏ sót sạt lở, giúp Recall đạt trên 77%."* |
| **"Hướng phát triển tiếp theo của đề tài là gì?"** | *"Em định hướng 3 bước: (1) Nâng cấp lên nnU-Net v2 tự động tối ưu hóa; (2) Chuyển từ phân đoạn nhị phân sang phân đoạn 4 cấp độ ưu tiên cứu nạn; (3) Tích hợp chỉ số mưa tích lũy suy giảm API và dữ liệu radar độ ẩm đất để giải quyết bài toán độ trễ sạt lở 10-15 ngày."* |

---
*Tài liệu được biên soạn phục vụ làm báo cáo thực tập, đồ án tốt nghiệp và nghiên cứu khoa học chuyên sâu.*
