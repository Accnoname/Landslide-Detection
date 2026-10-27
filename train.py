#!/usr/bin/env python3
"""
Main Training Entrypoint for Landslide Detection Project.

Usage:
    # 1. Chạy với file cấu hình YAML (Khuyên dùng)
    python train.py --config configs/baseline_resunet.yaml
    python train.py --config configs/fpn_resnet34.yaml
    python train.py --config configs/unetplusplus_resnet34.yaml

    # 2. Hoặc truyền tham số dòng lệnh trực tiếp
    python train.py --arch unet --encoder resnet34 --batch_size 16 --epochs 40
    python train.py --arch fpn --encoder resnet34 --batch_size 16 --epochs 30
    python train.py --arch unetplusplus --encoder resnet34 --batch_size 16 --epochs 30
"""
import os
import sys

# Ensure UTF-8 output on Windows consoles
for stream in [sys.stdout, sys.stderr]:
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8')
        except Exception:
            pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in [PROJECT_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.train import main

if __name__ == "__main__":
    main()
