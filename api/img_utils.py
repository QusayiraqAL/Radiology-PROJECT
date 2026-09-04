# -*- coding: utf-8 -*-
"""Shared image preprocessing — brain-region cropping.

Port of the notebook's `crop_brain_region` (Brain_Tumor_Hybrid_6Stage.ipynb): isolate the
brain by thresholding the dark background, clean with morphology, take the LARGEST connected
component, and crop to its bounding box with padding. Implemented with numpy/scipy so it needs
no OpenCV. Used identically in training and serving.
"""
import numpy as np
from PIL import Image
from scipy import ndimage


def crop_brain_region(pil_img, thresh=20, pad=10):
    """Return a PIL image cropped to the brain region. Falls back to the input if none found."""
    rgb = pil_img.convert("RGB")
    gray = np.asarray(rgb.convert("L"), dtype=np.uint8)
    mask = gray > thresh
    if mask.mean() < 0.01 or mask.mean() > 0.99:
        return rgb  # nearly all dark or all bright -> nothing meaningful to crop
    # morphological cleaning (open then close), like the notebook
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=2)
    lbl, n = ndimage.label(mask)
    if n == 0:
        return rgb
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    ys, xs = np.where(lbl == largest)
    if ys.size == 0:
        return rgb
    h, w = gray.shape
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)
    if (y1 - y0) < 16 or (x1 - x0) < 16:
        return rgb
    return rgb.crop((x0, y0, x1, y1))
