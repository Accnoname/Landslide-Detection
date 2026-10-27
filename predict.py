#!/usr/bin/env python3
"""
Landslide Mask Prediction and Inference Entrypoint.

Usage:
    python predict.py --checkpoint results/checkpoints/unet_resnet34_best.pth --test_dir ./data/processed/val/img --output_dir ./results/predictions
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in [PROJECT_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.predict_test import main

if __name__ == "__main__":
    main()
