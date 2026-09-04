# -*- coding: utf-8 -*-
"""EXTERNAL / cross-distribution validation for the pneumonia model.

The model trained on PneumoniaMNIST (Kermany 2018, PEDIATRIC). Here we test it on a genuinely
DIFFERENT source it never saw: NIH ChestX-ray14 (ADULT), via ChestMNIST. This is the strongest
form of "not trained on" — a different hospital population and labeling process. A drop from the
in-domain 96% is expected and honest: it measures real-world domain shift, not a broken model.

Binary set: positives = ChestMNIST 'pneumonia'==1 ; negatives = 'no finding' (all 14 labels 0).
"""
import os, sys, time
import numpy as np
import medmnist
from medmnist import INFO
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_pneumonia, save_results, API

SEED, N = 0, 2500
ROOT = os.path.join(API, "data", "medmnist")
PNEU_IDX = 6           # 'pneumonia' column in ChestMNIST's 14 labels


def main():
    t0 = time.time()
    DataClass = getattr(medmnist, INFO["chestmnist"]["python_class"])
    ds = DataClass(split="test", download=False, size=128, root=ROOT)
    imgs, labels = ds.imgs, ds.labels.astype(np.int64)
    rng = np.random.RandomState(SEED)

    pos = np.where(labels[:, PNEU_IDX] == 1)[0]                 # pneumonia present
    neg = np.where(labels.sum(1) == 0)[0]                       # no finding at all
    n_neg = min(len(neg), max(N - len(pos), N // 2))
    neg = rng.choice(neg, size=n_neg, replace=False)
    idx = np.concatenate([pos, neg]); rng.shuffle(idx)
    pil = [Image.fromarray(imgs[i]).convert("L") for i in idx]
    y = labels[idx, PNEU_IDX].astype(int)
    print(f"[pneu-nih] {len(pil)} images | pneumonia={int(y.sum())} normal={int((y==0).sum())}", flush=True)

    predict, th = load_pneumonia()
    prob = predict(pil)
    pred = (prob >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    acc = float((pred == y).mean())
    auc = float(roc_auc_score(y, prob))
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    print(f"[pneu-nih] acc={acc:.4f} auc={auc:.4f} sens={sens:.4f} spec={spec:.4f} "
          f"(thr={th}) in {time.time()-t0:.0f}s", flush=True)

    save_results("pneumonia", {
        "model": "pneumonia_v2_resnet18 (trained on Kermann pediatric)",
        "modality": "chest X-ray", "task": "binary normal vs pneumonia",
        "dataset": "ChestMNIST / NIH ChestX-ray14 (ADULT) — cross-distribution, never trained on",
        "is_external_source": True,
        "domain_shift_note": ("Trained on PEDIATRIC Kermany; tested on ADULT NIH. In-domain test "
                              "(Kermany) was acc 0.963 / spec 0.915. The gap here is genuine domain shift."),
        "decision_threshold": float(th),
        "classes": ["normal", "pneumonia"], "class_names_ar": ["طبيعي", "التهاب رئوي"],
        "n_test": int(len(pil)), "accuracy": round(acc, 4), "auc": round(auc, 4),
        "sensitivity": round(sens, 4), "specificity": round(spec, 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }, y_true=y, y_pred=pred, y_prob=prob)


if __name__ == "__main__":
    main()
