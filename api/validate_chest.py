# -*- coding: utf-8 -*-
"""
External validation of the pretrained CHEST model (TorchXRayVision DenseNet121)
on the held-out ChestMNIST TEST split (real NIH ChestX-ray14 images).
Computes real per-pathology ROC-AUC for the 14 shared labels.

ChestMNIST images are resized versions of NIH ChestX-ray14; we use size=224 so the
model runs at its native resolution. Reported AUC is a genuine measured number.
"""
import os
import json
import time
import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
from sklearn.metrics import roc_auc_score
import medmnist
from medmnist import INFO

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "data", "medmnist")
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 128          # download resolution (real NIH images); resized to 224 for the model
MAX_N = 6000        # cap number of test images evaluated on CPU (random, seeded)

# ChestMNIST label index -> TorchXRayVision pathology name
CHESTMNIST_LABELS = ["atelectasis", "cardiomegaly", "effusion", "infiltration", "mass",
                     "nodule", "pneumonia", "pneumothorax", "consolidation", "edema",
                     "emphysema", "fibrosis", "pleural", "hernia"]
MNIST_TO_XRV = {
    "atelectasis": "Atelectasis", "cardiomegaly": "Cardiomegaly", "effusion": "Effusion",
    "infiltration": "Infiltration", "mass": "Mass", "nodule": "Nodule",
    "pneumonia": "Pneumonia", "pneumothorax": "Pneumothorax", "consolidation": "Consolidation",
    "edema": "Edema", "emphysema": "Emphysema", "fibrosis": "Fibrosis",
    "pleural": "Pleural_Thickening", "hernia": "Hernia",
}
AR = {
    "Atelectasis": "انخماص رئوي", "Cardiomegaly": "تضخم القلب", "Effusion": "انصباب جنبي",
    "Infiltration": "ارتشاح رئوي", "Mass": "كتلة", "Nodule": "عقيدة رئوية",
    "Pneumonia": "التهاب رئوي", "Pneumothorax": "استرواح صدري", "Consolidation": "تصلّد رئوي",
    "Edema": "وذمة رئوية", "Emphysema": "نفاخ رئوي", "Fibrosis": "تليف رئوي",
    "Pleural_Thickening": "تسمّك جنبي", "Hernia": "فتق حجابي",
}


def main():
    t0 = time.time()
    DataClass = getattr(medmnist, INFO["chestmnist"]["python_class"])
    ds = DataClass(split="test", download=True, size=SIZE, root=DATA_ROOT)
    imgs = ds.imgs          # (N, SIZE, SIZE) uint8
    labels = ds.labels      # (N, 14) 0/1
    n_full = len(imgs)

    # cap to MAX_N real test images (random, seeded) to keep CPU eval feasible
    rng = np.random.RandomState(0)
    if n_full > MAX_N:
        sel = rng.choice(n_full, MAX_N, replace=False)
        imgs, labels = imgs[sel], labels[sel]
    n = len(imgs)
    print(f"[data] ChestMNIST test: using {n}/{n_full} images at {SIZE}px device={DEVICE}", flush=True)

    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(DEVICE).eval()
    xrv_names = list(model.pathologies)
    out_cols = {lab: xrv_names.index(MNIST_TO_XRV[lab]) for lab in CHESTMNIST_LABELS}

    resizer = xrv.datasets.XRayResizer(224)   # bring NIH images to the model's native size
    all_probs = np.zeros((n, 18), dtype=np.float32)
    bs = 32
    with torch.no_grad():
        for i in range(0, n, bs):
            batch = imgs[i:i+bs].astype(np.float32)
            batch = xrv.datasets.normalize(batch, 255)
            resized = np.stack([resizer(im[None, ...])[0] for im in batch])  # (b,224,224)
            t = torch.from_numpy(resized[:, None, :, :]).to(DEVICE)
            out = model(t).cpu().numpy()
            all_probs[i:i+bs] = out
            if i % (bs * 20) == 0:
                print(f"  {i}/{n}", flush=True)

    per_label = {}
    aucs = []
    for j, lab in enumerate(CHESTMNIST_LABELS):
        y = labels[:, j].astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        p = all_probs[:, out_cols[lab]]
        auc = roc_auc_score(y, p)
        xrvname = MNIST_TO_XRV[lab]
        per_label[xrvname] = {
            "name_ar": AR[xrvname], "auc": round(float(auc), 4),
            "positives": int(y.sum()), "n": int(len(y)),
        }
        aucs.append(auc)

    metrics = {
        "model": "chest_xray_densenet121_torchxrayvision",
        "task": "14-label chest X-ray pathology (external validation)",
        "modality": "chest X-ray",
        "eval_dataset": f"ChestMNIST test split (MedMNIST v2; NIH ChestX-ray14), {SIZE}px source",
        "n_test": int(n),
        "n_test_full": int(n_full),
        "eval_resolution": f"{SIZE}px resized to 224 (model native)",
        "mean_auc": round(float(np.mean(aucs)), 4),
        "per_label_auc": per_label,
        "note": f"Pretrained model externally validated on {n} real NIH ChestX-ray14 test images "
                f"({SIZE}px upscaled to 224). AUC computed per label where both classes present. "
                "This is a conservative real measurement (source images downsampled by MedMNIST).",
        "validated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eval_seconds": round(time.time() - t0, 1),
        "device": DEVICE,
    }
    with open(os.path.join(MODEL_DIR, "chest_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("\n[RESULT] mean_auc =", metrics["mean_auc"])
    for k, v in per_label.items():
        print(f"  {v['auc']:.4f}  {k} ({v['name_ar']})  pos={v['positives']}")
    print("CHEST_VALIDATE_DONE")


if __name__ == "__main__":
    main()
