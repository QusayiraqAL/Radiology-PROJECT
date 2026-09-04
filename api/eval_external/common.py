# -*- coding: utf-8 -*-
"""Shared helpers for EXTERNAL validation — testing each model on data it was NOT trained on.

Loads the SAME served checkpoints with the SAME preprocessing as main.py (so the numbers
here are what the deployed API would actually produce), plus dHash de-duplication so we can
prove a test image is genuinely unseen rather than a near-copy of a training image.
"""
import os, sys, json, io
import numpy as np
from PIL import Image

HERE = os.path.dirname(__file__)
API = os.path.dirname(HERE)
sys.path.insert(0, API)                      # import the model code from api/
MODELS = os.path.join(API, "models")
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

import torch
from brain_split import dhash_bits            # reuse the exact hash from the leakage work

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
_IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ----------------------------- de-duplication -----------------------------
def hash_matrix(pil_list):
    return np.stack([dhash_bits(im) for im in pil_list]).astype(np.int16)


def min_hamming_to_ref(query_bits, ref_bits, chunk=256):
    """Min Hamming distance from each query hash to any reference hash (vectorized)."""
    sa = query_bits.sum(1, keepdims=True)
    sb = ref_bits.sum(1, keepdims=True).T
    out = np.empty(len(query_bits), dtype=np.int16)
    for i in range(0, len(query_bits), chunk):
        d = sa[i:i+chunk] + sb - 2 * (query_bits[i:i+chunk] @ ref_bits.T)
        out[i:i+chunk] = d.min(1)
    return out


# ----------------------------- brain (v2 cropped) -----------------------------
def load_brain(ckpt="brain_tumor_mri_v2.pt"):
    from nets import build_brain_resnet
    from img_utils import crop_brain_region
    ck = torch.load(os.path.join(MODELS, ckpt), map_location="cpu", weights_only=False)
    classes, size, cropped = ck["classes"], ck.get("size", 128), ck.get("cropped", True)
    net = build_brain_resnet(len(classes), pretrained=False, dropout=ck.get("dropout", 0.0)).eval()
    net.load_state_dict(ck["state_dict"])

    @torch.no_grad()
    def predict(pil_list, bs=64):
        probs = []
        for i in range(0, len(pil_list), bs):
            xs = []
            for im in pil_list[i:i+bs]:
                im2 = im.convert("RGB")
                if cropped:
                    im2 = crop_brain_region(im2)          # SAME as serving
                im2 = im2.resize((size, size))
                x = torch.from_numpy(np.asarray(im2, np.float32) / 255.0).permute(2, 0, 1)
                xs.append((x - _IM_MEAN) / _IM_STD)
            probs.append(torch.softmax(net(torch.stack(xs)), 1).numpy())
        return np.concatenate(probs)
    return predict, classes


# ----------------------------- pneumonia (v2 resnet @224) -----------------------------
def load_pneumonia():
    from nets import build_pneumonia_resnet
    ck = torch.load(os.path.join(MODELS, "pneumonia_v2.pt"), map_location="cpu", weights_only=False)
    size, th = ck.get("size", 224), ck.get("threshold", 0.5)
    net = build_pneumonia_resnet(2, pretrained=False, dropout=ck.get("dropout", 0.3)).eval()
    net.load_state_dict(ck["state_dict"])

    @torch.no_grad()
    def predict(pil_list, bs=48):
        out = []
        for i in range(0, len(pil_list), bs):
            xs = []
            for im in pil_list[i:i+bs]:
                im2 = im.convert("RGB").resize((size, size))
                x = torch.from_numpy(np.asarray(im2, np.float32) / 255.0).permute(2, 0, 1)
                xs.append((x - _IM_MEAN) / _IM_STD)
            out.append(torch.softmax(net(torch.stack(xs)), 1)[:, 1].numpy())
        return np.concatenate(out)
    return predict, th


# ----------------------------- chest (TorchXRayVision, pretrained) -----------------------------
def load_chest():
    import torchvision, torchxrayvision as xrv
    model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
    tf = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    pathologies = list(model.pathologies)

    @torch.no_grad()
    def predict(pil_list, bs=32):
        out = []
        for i in range(0, len(pil_list), bs):
            xs = []
            for im in pil_list[i:i+bs]:
                a = np.asarray(im.convert("L"), np.float32)
                a = xrv.datasets.normalize(a, 255)[None, ...]
                xs.append(torch.from_numpy(tf(a)))
            out.append(model(torch.stack(xs)).numpy())
        return np.concatenate(out)
    return predict, pathologies


# ----------------------------- save -----------------------------
def save_results(name, meta, y_true=None, y_pred=None, y_prob=None, extra_arrays=None):
    arrs = {}
    if y_true is not None: arrs["y_true"] = np.asarray(y_true)
    if y_pred is not None: arrs["y_pred"] = np.asarray(y_pred)
    if y_prob is not None: arrs["y_prob"] = np.asarray(y_prob)
    for k, v in (extra_arrays or {}).items():
        arrs[k] = np.asarray(v)
    np.savez_compressed(os.path.join(RESULTS, f"{name}.npz"), **arrs)
    json.dump(meta, open(os.path.join(RESULTS, f"{name}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[saved] results/{name}.npz + .json  ({meta.get('n_test')} samples)", flush=True)
