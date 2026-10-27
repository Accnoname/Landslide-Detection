# Landslide Detection in Multi-Source Satellite Imagery
## Benchmark Reproduction and Optimization on Landslide4Sense (IEEE TGRS 2022)

**Author:** Accnoname  
**Repository:** https://github.com/Accnoname/Landslide-Detection  
**Target Dataset:** Landslide4Sense (14 channels: Sentinel-2 Multispectral + ALOS PALSAR DEM and Slope)  
**Reference Paper:** IEEE Transactions on Geoscience and Remote Sensing (TGRS 2022)

---

## 1. Overview and Key Achievements

This repository implements a high-performance deep learning pipeline for automated landslide semantic segmentation using 14-band multi-source satellite imagery. The models are evaluated on the Landslide4Sense benchmark dataset comprising Sentinel-2 multispectral imagery and ALOS PALSAR topographic data (Elevation and Slope) at 10-meter spatial resolution.

### Primary Results Summary:
* **F1-Score:** Achieved **75.77%**, surpassing the IEEE TGRS 2022 published baseline of **71.65%** (+4.12% absolute improvement).
* **Intersection over Union (IoU):** Achieved **60.99%**, outperforming the reference benchmark of **55.83%** (+5.16% absolute improvement).
* **Recall (Detection Coverage):** Reached **77.49%**, significantly reducing false negatives compared to the published benchmark's **66.80%** (+10.69% improvement in disaster target recovery).
* **Inference Speed:** ResU-Net processes at **266.9 FPS**, while DeepLabV3+ reaches **644.8 FPS** on an NVIDIA GPU.

---

## 2. Experimental Benchmark Results

Quantitative evaluation conducted on an independent validation split of 245 multi-spectral patches ($128 \times 128 \times 14$ channels):

| Model Architecture | Parameters | Precision | Recall | F1-Score | IoU | Inference FPS | Role in Project |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **U-Net (ResNet34 Backbone / ResU-Net)** | **24.5M** | **74.21%** | **77.49%** | **75.77%** | **60.99%** | 266.9 FPS | Primary Champion Model (U-Net architecture with ResNet34 encoder) |
| **DeepLabV3+ (ResNet34)** | 22.5M | 66.42% | 77.03% | 71.33% | 55.44% | **644.8 FPS** | Comparative Baseline (Large-scale rapid screening) |
| **U-Net + TTA (8-Fold Inference)** | 24.5M | **77.58%** | 73.91% | 75.70% | 60.90% | 34.5 FPS | High-Precision Inference (Edge noise suppression) |
| **FPN (Feature Pyramid Network / ResNet34)** | 23.2M | 68.29% | 58.92% | 63.26% | 46.27% | 312.4 FPS | Comparative Multi-Scale Model (Addition-based merging) |
| **Published Benchmark ResU-Net (IEEE TGRS)** | — | 77.20% | 66.80% | 71.65% | 55.83% | — | Reference Benchmark Target (IEEE TGRS 2022) |
| **Published Standard Plain U-Net (IEEE TGRS)** | — | ~70.5% | ~66.0% | 68.50% | 52.09% | — | Standard Conventional U-Net Baseline |

### Technical Analysis:
1. **U-Net vs. FPN Architectural Discrepancy:** U-Net outperforms FPN significantly (+12.51% F1 and +14.72% IoU). While FPN employs element-wise addition across downscaled pyramid levels (which induces feature blurring along fractured landslide scar boundaries), U-Net maintains full-depth channel concatenation (`torch.cat`), preserving uncompromised radiometric and topographic gradient details (ALOS DEM and Slope) across resolution scales.
2. **U-Net Encoder-Decoder Paradigm:** The model strictly adheres to the U-Net architecture (symmetrical Contracting Encoder and Expanding Decoder paths). By employing a ResNet-34 backbone as the encoder (commonly termed ResU-Net), residual skip connections mitigate vanishing gradients while full-depth long skip connections route fine spatial contours directly to the decoder.
3. **Combo Loss Strategy:** Extreme class imbalance (landslides occupy less than 3% of spatial coverage) is effectively addressed using a composite objective:
   $$\mathcal{L}_{\text{total}} = 0.6 \cdot \mathcal{L}_{\text{Dice}} + 0.4 \cdot \mathcal{L}_{\text{BCE}}$$
   The Dice component heavily penalizes false negatives, lifting Recall from 66.80% to 77.49%.
4. **Mathematical Relationship between IoU and F1:**
   $$\text{IoU} = \frac{F_1}{2 - F_1} \quad \Longleftrightarrow \quad F_1 = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$$
   An IoU of 60.99% corresponds algebraically to an F1-Score of 75.77%. In geospatial segmentation, an IoU above 60% represents top-tier overlap accuracy.

---

## 3. Dataset Specification

The Landslide4Sense dataset provides 14 co-registered raster channels per sample:

| Band Index | Sensor / Source | Spectral / Physical Property | Spatial Resolution | Central Wavelength / Unit |
| :---: | :--- | :--- | :---: | :---: |
| 1 | Sentinel-2 | B01 - Coastal Aerosols | 60 m | 443 nm |
| 2 | Sentinel-2 | B02 - Blue | 10 m | 490 nm |
| 3 | Sentinel-2 | B03 - Green | 10 m | 560 nm |
| 4 | Sentinel-2 | B04 - Red | 10 m | 665 nm |
| 5 | Sentinel-2 | B05 - Vegetation Red Edge 1 | 20 m | 705 nm |
| 6 | Sentinel-2 | B06 - Vegetation Red Edge 2 | 20 m | 740 nm |
| 7 | Sentinel-2 | B07 - Vegetation Red Edge 3 | 20 m | 783 nm |
| 8 | Sentinel-2 | B08 - Broad NIR | 10 m | 842 nm |
| 9 | Sentinel-2 | B08A - Narrow NIR | 20 m | 865 nm |
| 10 | Sentinel-2 | B09 - Water Vapour | 60 m | 945 nm |
| 11 | Sentinel-2 | B11 - SWIR 1 | 20 m | 1610 nm |
| 12 | Sentinel-2 | B12 - SWIR 2 | 20 m | 2190 nm |
| 13 | ALOS PALSAR | Slope (Terrain Gradient) | 12.5 m | Degrees |
| 14 | ALOS PALSAR | Elevation (Digital Elevation Model - DEM) | 12.5 m | Meters above sea level |

---

## 4. Cloud Execution on Google Colab

A dedicated notebook (`run_on_colab.ipynb`) is provided for one-click training on a cloud GPU (Tesla T4) with automated Google Drive checkpoint synchronization.

Direct launcher URL:  
https://colab.research.google.com/github/Accnoname/Landslide-Detection/blob/main/run_on_colab.ipynb

### Automated Pipeline Workflow:
1. **GPU Allocation:** Assign a standard T4 GPU via `Runtime -> Change runtime type -> T4 GPU`.
2. **Drive Mount:** Checkpoints and logs are written directly to `/content/drive/MyDrive/Detect_Results/`.
3. **Direct Data Fetch:** Downloads the Landslide4Sense dataset directly from Zenodo servers at gigabit speeds without requiring Kaggle API credentials.
4. **Execution:** Runs `src/train.py` with batch size 32, completing 20 epochs in under 4 minutes.

---

## 5. Local Setup and Execution

### 5.1. Environment Installation
```bash
# Clone the repository
git clone https://github.com/Accnoname/Landslide-Detection.git
cd Landslide-Detection

# Install required dependencies
pip install -r requirements.txt
```

### 5.2. Data Preparation
```bash
# Automated download and directory structuring from Zenodo
python src/download_data.py

# Generate independent validation split (245 labeled pairs)
python src/create_val_split.py
```

### 5.3. Training
* **Train ResU-Net Standard Pipeline via YAML Configuration:**
  ```bash
  python train.py --config configs/baseline_resunet.yaml
  ```

* **Train Multi-Scale Feature Pyramid Network (FPN):**
  ```bash
  python train.py --config configs/fpn_resnet34.yaml
  ```

* **Train Nested Dense Skip Connections (UNet++):**
  ```bash
  python train.py --config configs/unetplusplus_resnet34.yaml
  ```

* **Train with 2D Data Augmentation and In-Memory RAM Caching:**
  ```bash
  python train.py --arch unet --encoder resnet34 --batch_size 32 --epochs 20 --augment --cache_data
  ```

### 5.4. Evaluation and Optimization
* **Standard Evaluation and Metric Computation:**
  ```bash
  python evaluate.py --checkpoint results/checkpoints/unet_resnet34_best.pth
  ```

* **Decision Threshold Sensitivity Analysis (0.20 to 0.80):**
  ```bash
  python evaluate.py --checkpoint results/checkpoints/unet_resnet34_best.pth --tune_threshold
  ```

* **Test-Time Augmentation (8-Fold Averaging):**
  ```bash
  python evaluate.py --checkpoint results/checkpoints/unet_resnet34_best.pth --tta
  ```

* **Inference on Test Patches & Packaging Submission Archive:**
  ```bash
  python predict.py --checkpoint results/checkpoints/unet_resnet34_best.pth
  ```

---

## 6. Metric Definitions and Operational Modes

All metrics are evaluated at the global pixel level (micro-averaged over all validation samples):

* **Precision:** $\frac{TP}{TP + FP}$  
  Measures false alarm suppression. High precision ensures operational resources are not deployed to false targets such as ploughed farmland or dry riverbeds.
* **Recall:** $\frac{TP}{TP + FN}$  
  Measures disaster target detection completeness. In disaster response, minimizing false negatives takes precedence to ensure affected areas are not overlooked.
* **F1-Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$  
  Harmonic mean penalizing disproportionate models, serving as the primary benchmark criterion.
* **IoU (Jaccard Index):** $\frac{TP}{TP + FP + FN}$  
  Geometric overlap ratio between predicted masks and ground-truth delineations, critical for spatial area estimation ($m^2$).

### Operational Regimes via Threshold Calibration:
* **Emergency Relief Mode ($Th = 0.20$):** Recall reaches **81.55%** ($F_1 = 75.20\%$), prioritizing life safety by sweeping all potential scars.
* **Standard Benchmark Mode ($Th = 0.50$):** Peak overall performance at **$F_1 = 75.77\%$** and **$\text{IoU} = 60.99\%$**.
* **Land Administration Mode ($Th = 0.80$):** Precision reaches **78.20%** ($F_1 = 75.17\%$), guaranteeing confirmed targets before civil engineering interventions.

---

## 7. Repository Layout

```text
Landslide-Detection/
├── configs/                             # Model hyperparameter configurations (YAML)
│   ├── baseline_resunet.yaml
│   ├── fpn_resnet34.yaml
│   └── unetplusplus_resnet34.yaml
├── docs/                                # Technical reports and guides
│   └── TRAIN.md                         # End-to-end 7-step training mechanics guide
├── src/                                 # Core source package
│   ├── data/
│   │   ├── dataset.py                   # 14-band DataLoader with RAM caching and 2D transforms
│   │   └── download.py                  # Direct Zenodo dataset downloader
│   ├── models/
│   │   └── builder.py                   # Model architecture factory (U-Net, FPN, UNet++, DeepLabV3+)
│   ├── utils/
│   │   ├── losses.py                    # Composite ComboLoss (Dice + BCE)
│   │   └── metrics.py                   # Pixel-level confusion matrix metric computation
│   ├── postprocess/
│   │   ├── tune_threshold.py            # Decision threshold calibration utility
│   │   └── evaluate_tta.py              # 8-fold Test-Time Augmentation evaluator
│   └── experiments/                     # Specialized experimental benchmarks
│       ├── create_val_split.py          # Validation cohort partitioner
│       ├── compare_models.py            # Cross-architecture benchmarking script
│       └── nnunet/                      # Automated nnU-Net v2 pre-processing & training
├── train.py                             # Root execution entrypoint for model training
├── evaluate.py                          # Root execution entrypoint for evaluation and TTA
├── predict.py                           # Root execution entrypoint for test inference
├── requirements.txt                     # Python package dependencies
├── run_on_colab.ipynb                   # Executable cloud launcher notebook
├── RESEARCH_NOTES.md                    # Research methodology and defense argumentation
├── ROADMAP.md                           # Multi-phase project roadmap
├── .gitignore                           # Excludes heavy data, weights, and private credentials
└── README.md                            # Comprehensive technical documentation
```

---

## 8. References

1. Ghorbanzadeh, O. et al. (2022). "Landslide4Sense: Reference Benchmark Data and Deep Learning Models for Landslide Detection." *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*, 60, 1-17.
2. Isensee, F. et al. (2021). "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." *Nature Methods*, 18(2), 203-211.
3. Chen, L. C. et al. (2018). "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation." *Proceedings of the European Conference on Computer Vision (ECCV)*.

---
*Maintained by **Accnoname**. For issues, questions, or extensions, please open an Issue on the GitHub repository.*
