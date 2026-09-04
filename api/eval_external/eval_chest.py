# -*- coding: utf-8 -*-
"""EXTERNAL validation for the chest 14-label model (TorchXRayVision DenseNet121).

The model is pretrained (never trained on MedMNIST). We evaluate on a FRESH 2500-image slice
of the ChestMNIST test split (NIH ChestX-ray14, 128px). Multi-label task -> the honest metric
is per-label ROC-AUC (threshold-free). For a confusion-matrix view we also binarize into
"any finding vs no finding".
"""
import os, sys, time
import numpy as np
import medmnist
from medmnist import INFO
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_chest, save_results, API

SEED, N = 0, 2500
ROOT = os.path.join(API, "data", "medmnist")
# ChestMNIST label order (14) -> TorchXRayVision pathology name
MNIST14 = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
           "Pleural_Thickening", "Hernia"]
AR = {"Atelectasis": "انخماص", "Cardiomegaly": "تضخم القلب", "Effusion": "انصباب جنبي",
      "Infiltration": "ارتشاح", "Mass": "كتلة", "Nodule": "عقيدة", "Pneumonia": "التهاب رئوي",
      "Pneumothorax": "استرواح صدري", "Consolidation": "تصلّد", "Edema": "وذمة",
      "Emphysema": "نفاخ", "Fibrosis": "تليّف", "Pleural_Thickening": "تسمّك جنبي", "Hernia": "فتق"}


def main():
    t0 = time.time()
    DataClass = getattr(medmnist, INFO["chestmnist"]["python_class"])
    ds = DataClass(split="test", download=False, size=128, root=ROOT)
    imgs, labels = ds.imgs, ds.labels.astype(np.int64)     # (22433,128,128), (22433,14)
    rng = np.random.RandomState(SEED)
    # fresh 2500 slice NOT among the first 6000 used in the prior internal validation
    pool = np.arange(6000, len(imgs))
    idx = rng.choice(pool, size=min(N, len(pool)), replace=False)
    pil = [Image.fromarray(imgs[i]).convert("L") for i in idx]
    Y = labels[idx]
    print(f"[chest] evaluating {len(pil)} fresh ChestMNIST test images", flush=True)

    predict, pathologies = load_chest()
    prob = predict(pil)                                    # (N, 18) model pathologies
    p_index = {p: j for j, p in enumerate(pathologies)}

    per_label = {}
    for k, name in enumerate(MNIST14):
        j = p_index.get(name)
        if j is None:
            continue
        yk = Y[:, k]
        if yk.sum() == 0 or yk.sum() == len(yk):
            continue
        per_label[name] = {"auc": round(float(roc_auc_score(yk, prob[:, j])), 4),
                           "positives": int(yk.sum()), "name_ar": AR.get(name, name)}
    mean_auc = round(float(np.mean([v["auc"] for v in per_label.values()])), 4)

    # confusion-matrix view: "any finding vs no finding"
    true_any = (Y.sum(1) > 0).astype(int)
    model_cols = [p_index[n] for n in MNIST14 if n in p_index]
    pred_any = (prob[:, model_cols].max(1) >= 0.5).astype(int)
    acc_any = float((true_any == pred_any).mean())
    print(f"[chest] mean_auc={mean_auc} over {len(per_label)} labels | any-finding acc={acc_any:.4f} "
          f"in {time.time()-t0:.0f}s", flush=True)

    save_results("chest", {
        "model": "chest_xray_densenet121_torchxrayvision (pretrained)",
        "modality": "chest X-ray", "task": "14-label multi-label pathology",
        "dataset": "ChestMNIST test (NIH ChestX-ray14, 128px) — fresh 2500 slice, indices 6000+",
        "is_external_source": True,
        "note": "Model is pretrained and never saw MedMNIST; this slice also excludes the prior 6000-image internal eval.",
        "metric_primary": "per-label ROC-AUC (multi-label; confusion matrix is the any-finding binarization)",
        "classes": ["no_finding", "finding"], "class_names_ar": ["لا مؤشر", "يوجد مؤشر"],
        "per_label_auc": per_label, "mean_auc": mean_auc,
        "n_test": int(len(pil)), "accuracy": round(acc_any, 4),
    }, y_true=true_any, y_pred=pred_any, y_prob=prob[:, model_cols].max(1))


if __name__ == "__main__":
    main()
