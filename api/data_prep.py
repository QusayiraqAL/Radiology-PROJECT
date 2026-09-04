# -*- coding: utf-8 -*-
"""
Download REAL medical imaging datasets (MedMNIST v2) with official train/val/test splits.
Data source: Yang et al., "MedMNIST v2", Scientific Data 2023 (Zenodo-hosted, real clinical images).
"""
import os
import numpy as np
import medmnist
from medmnist import INFO

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data", "medmnist")
os.makedirs(DATA_ROOT, exist_ok=True)

DATASETS = ["pneumoniamnist", "chestmnist", "breastmnist"]

def main():
    for flag in DATASETS:
        info = INFO[flag]
        DataClass = getattr(medmnist, info["python_class"])
        print(f"\n=== {flag} ===")
        print("task:", info["task"])
        print("label:", info["label"])
        splits = {}
        for split in ["train", "val", "test"]:
            ds = DataClass(split=split, download=True, size=28, root=DATA_ROOT)
            splits[split] = (ds.imgs.shape, ds.labels.shape)
            print(f"  {split}: imgs={ds.imgs.shape} labels={ds.labels.shape} "
                  f"dtype={ds.imgs.dtype} min={ds.imgs.min()} max={ds.imgs.max()}")
    print("\nALL_DOWNLOADED_OK")

if __name__ == "__main__":
    main()
