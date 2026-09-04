# -*- coding: utf-8 -*-
"""
Train a REAL brain-tumor MRI classifier on the standard 4-class Brain Tumor MRI dataset
(real brain MRI slices: glioma / meningioma / notumor / pituitary), loaded from a single
HuggingFace parquet. Fine-tunes an ImageNet-pretrained ResNet-18.

We build a stratified, seeded train/val/test split (70/15/15) and report metrics on the
held-out TEST split (unseen during training).
"""
import os
import io
import json
import time
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             precision_score, recall_score, f1_score)

from nets import build_brain_resnet

HERE = os.path.dirname(__file__)
PARQUET = os.path.join(HERE, "data", "brain_parquet", "data", "train-00000-of-00001.parquet")
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SIZE = 128
SEED = 0
EPOCHS = 10
BATCH = 32
LR = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def class_names_from_schema(schema, label_col):
    """Extract ClassLabel names from HF parquet schema metadata; fallback to standard order."""
    try:
        meta = schema.metadata or {}
        for k, v in meta.items():
            if b"huggingface" in k.lower() or b"class" in v.lower():
                info = json.loads(v.decode("utf-8"))
                feats = info.get("info", {}).get("features", info.get("features", {}))
                f = feats.get(label_col)
                if isinstance(f, dict) and "names" in f:
                    return f["names"]
    except Exception:
        pass
    return ["glioma", "meningioma", "notumor", "pituitary"]


def load_parquet():
    t = pq.read_table(PARQUET)
    cols = t.column_names
    print("[parquet] columns:", cols)
    # detect label column
    label_col = next((c for c in cols if c.lower() in ("label", "labels", "class")), cols[-1])
    img_col = next((c for c in cols if c.lower() in ("image", "img", "images")), cols[0])
    classes = class_names_from_schema(t.schema, label_col)
    df = t.to_pydict()
    raw_imgs = df[img_col]
    labels = np.array(df[label_col], dtype=np.int64)

    def to_bytes(cell):
        if isinstance(cell, dict):
            return cell.get("bytes") or cell.get("path")
        return cell
    img_bytes = [to_bytes(c) for c in raw_imgs]
    print(f"[parquet] n={len(labels)} classes={classes} "
          f"label_range=({labels.min()},{labels.max()})")
    return img_bytes, labels, classes


class BrainDS(Dataset):
    def __init__(self, img_bytes, labels, idx, tf):
        self.img_bytes = img_bytes; self.labels = labels; self.idx = idx; self.tf = tf

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        im = Image.open(io.BytesIO(self.img_bytes[j])).convert("RGB")
        return self.tf(im), int(self.labels[j])


def stratified_split(labels, seed=SEED):
    rng = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx); n_te = int(0.15 * n); n_va = int(0.15 * n)
        te += idx[:n_te].tolist()
        va += idx[n_te:n_te + n_va].tolist()
        tr += idx[n_te + n_va:].tolist()
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te


def main():
    t0 = time.time()
    img_bytes, labels, classes = load_parquet()
    tr_idx, va_idx, te_idx = stratified_split(labels)
    print(f"[split] train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)} device={DEVICE}")

    train_tf = transforms.Compose([
        transforms.Grayscale(3), transforms.Resize((SIZE, SIZE)),
        transforms.RandomRotation(10), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(3), transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    tr_loader = DataLoader(BrainDS(img_bytes, labels, tr_idx, train_tf), batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(BrainDS(img_bytes, labels, va_idx, eval_tf), batch_size=64)
    te_loader = DataLoader(BrainDS(img_bytes, labels, te_idx, eval_tf), batch_size=64)

    counts = np.bincount([labels[i] for i in tr_idx], minlength=len(classes))
    w = counts.sum() / (len(classes) * np.maximum(counts, 1))
    class_weight = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    print(f"[data] train class counts={counts.tolist()} weights={w.round(3).tolist()}")

    model = build_brain_resnet(num_classes=len(classes), pretrained=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit = nn.CrossEntropyLoss(weight=class_weight)

    @torch.no_grad()
    def collect(loader):
        model.eval(); ys, ps = [], []
        for xb, yb in loader:
            p = torch.softmax(model(xb.to(DEVICE)), 1).cpu().numpy()
            ps.append(p); ys.append(yb.numpy())
        return np.concatenate(ys), np.concatenate(ps)

    best_acc, best_state = 0.0, None
    for ep in range(EPOCHS):
        model.train(); tot = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        yv, pv = collect(va_loader)
        va_acc = accuracy_score(yv, pv.argmax(1))
        print(f"  epoch {ep+1:2d}/{EPOCHS} loss={tot/len(tr_idx):.4f} val_acc={va_acc:.4f}", flush=True)
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    yt, pt = collect(te_loader)
    pred = pt.argmax(1)
    try:
        auc_macro = roc_auc_score(yt, pt, multi_class="ovr", average="macro")
    except Exception:
        auc_macro = float("nan")
    per_class = {}
    for i, c in enumerate(classes):
        yb = (yt == i).astype(int); pb = (pred == i).astype(int)
        per_class[c] = {
            "sensitivity_recall": round(recall_score(yb, pb, zero_division=0), 4),
            "precision": round(precision_score(yb, pb, zero_division=0), 4),
            "f1": round(f1_score(yb, pb, zero_division=0), 4),
            "support": int((yt == i).sum()),
        }
    metrics = {
        "model": "brain_tumor_mri_resnet18",
        "task": f"{len(classes)}-class tumor classification",
        "modality": "brain MRI",
        "dataset": "Brain Tumor MRI Dataset (HuggingFace Hemg/Brain-Tumor-MRI-Dataset) — real brain MRI slices",
        "split": "stratified seeded 70/15/15 (held-out test)",
        "input_size": SIZE,
        "classes": classes,
        "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx),
        "test_accuracy": round(accuracy_score(yt, pred), 4),
        "test_auc_macro_ovr": None if np.isnan(auc_macro) else round(float(auc_macro), 4),
        "test_macro_f1": round(f1_score(yt, pred, average="macro"), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(yt, pred).tolist(),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": round(time.time() - t0, 1),
        "device": DEVICE,
    }
    torch.save({"state_dict": model.state_dict(), "size": SIZE, "classes": classes},
               os.path.join(MODEL_DIR, "brain_tumor_mri.pt"))
    with open(os.path.join(MODEL_DIR, "brain_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("\n[RESULT]", json.dumps(metrics, ensure_ascii=False))
    print("BRAIN_TRAIN_DONE")


if __name__ == "__main__":
    main()
