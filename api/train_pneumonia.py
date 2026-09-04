# -*- coding: utf-8 -*-
"""
Train a REAL pneumonia classifier on PneumoniaMNIST (pediatric chest X-rays,
Kermany et al. 2018, packaged by MedMNIST v2). Official train/val/test splits.
Saves trained weights + a validation report computed on the held-out TEST split.
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score)
import medmnist
from medmnist import INFO

from nets import SmallXRayCNN

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "data", "medmnist")
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SIZE = 64
SEED = 0
EPOCHS = 20
BATCH = 128
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_split(split):
    DataClass = getattr(medmnist, INFO["pneumoniamnist"]["python_class"])
    ds = DataClass(split=split, download=True, size=SIZE, root=DATA_ROOT)
    x = ds.imgs.astype(np.float32) / 255.0          # (N, H, W)
    x = (x - 0.5) / 0.5                              # normalize to [-1, 1]
    x = x[:, None, :, :]                            # (N, 1, H, W)
    y = ds.labels.astype(np.float32).reshape(-1)    # (N,)
    return torch.from_numpy(x), torch.from_numpy(y)


def main():
    t0 = time.time()
    xtr, ytr = load_split("train")
    xva, yva = load_split("val")
    xte, yte = load_split("test")
    print(f"[data] train={len(xtr)} val={len(xva)} test={len(xte)} size={SIZE} device={DEVICE}")

    # class imbalance handling
    pos = float(ytr.sum()); neg = float(len(ytr) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1)], device=DEVICE)

    tr_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=BATCH, shuffle=True)
    model = SmallXRayCNN(num_classes=1, in_ch=1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def evaluate(x, y):
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(x), 256):
                logit = model(x[i:i+256].to(DEVICE)).squeeze(1)
                probs.append(torch.sigmoid(logit).cpu().numpy())
        p = np.concatenate(probs)
        return p, roc_auc_score(y.numpy(), p)

    best_auc, best_state = 0.0, None
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb).squeeze(1), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        _, va_auc = evaluate(xva, yva)
        print(f"  epoch {ep+1:2d}/{EPOCHS} loss={tot/len(xtr):.4f} val_auc={va_auc:.4f}")
        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # ---- final evaluation on held-out TEST split (choose threshold on val) ----
    pva, _ = evaluate(xva, yva)
    # pick threshold maximizing Youden's J on validation
    ths = np.linspace(0.05, 0.95, 91)
    yv = yva.numpy()
    best_th, best_j = 0.5, -1
    for th in ths:
        pred = (pva >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(yv, pred, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        j = sens + spec - 1
        if j > best_j:
            best_j, best_th = j, float(th)

    pte, te_auc = evaluate(xte, yte)
    yt = yte.numpy()
    pred = (pte >= best_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    metrics = {
        "model": "pneumonia_xray_smallcnn",
        "task": "binary (normal vs pneumonia)",
        "modality": "chest X-ray",
        "dataset": "PneumoniaMNIST (MedMNIST v2; Kermany 2018 pediatric CXR)",
        "input_size": SIZE,
        "n_train": len(xtr), "n_val": len(xva), "n_test": len(xte),
        "decision_threshold": round(best_th, 3),
        "test_accuracy": round(accuracy_score(yt, pred), 4),
        "test_auc": round(float(te_auc), 4),
        "test_sensitivity_recall": round(recall_score(yt, pred), 4),
        "test_specificity": round(tn / max(tn + fp, 1), 4),
        "test_precision": round(precision_score(yt, pred, zero_division=0), 4),
        "test_f1": round(f1_score(yt, pred), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "labels": {"0": "normal", "1": "pneumonia"},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": round(time.time() - t0, 1),
        "device": DEVICE,
    }
    torch.save({"state_dict": model.state_dict(), "size": SIZE, "threshold": best_th},
               os.path.join(MODEL_DIR, "pneumonia_xray.pt"))
    with open(os.path.join(MODEL_DIR, "pneumonia_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("\n[RESULT]", json.dumps(metrics, ensure_ascii=False))
    print("PNEUMONIA_TRAIN_DONE")


if __name__ == "__main__":
    main()
